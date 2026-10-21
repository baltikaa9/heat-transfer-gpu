#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define DIM 2048
#define STEPS 90
#define SPEED 0.25f
#define MAX_TEMP 1.0f
#define MIN_TEMP 0.0001f

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t _e = (call);                                              \
        if (_e != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(_e));              \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

__global__ void blend_kernel(
    float* __restrict__ outSrc,
    const float* __restrict__ inSrc,
    int width,
    int local_h
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y < 1 || y > local_h) return;

    int offset = x + y * width;
    float v = inSrc[offset];
    float top = inSrc[offset - width];
    float bot = inSrc[offset + width];
    float left = inSrc[offset - 1];
    float right = inSrc[offset + 1];

    outSrc[offset] = v + SPEED * (top + bot + left + right - 4.0f * v);
}

static bool p2pAvailable(int workers) {
    for (int source = 0; source < workers; source++) {
        for (int target = 0; target < workers; target++) {
            if (source == target) continue;
            int canAccess = 0;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccess, source, target));
            if (!canAccess) return false;
        }
    }
    return true;
}

static void enableP2P(int workers) {
    for (int source = 0; source < workers; source++) {
        CUDA_CHECK(cudaSetDevice(source));
        for (int target = 0; target < workers; target++) {
            if (source == target) continue;
            cudaError_t error = cudaDeviceEnablePeerAccess(target, 0);
            if (error != cudaSuccess && error != cudaErrorPeerAccessAlreadyEnabled) {
                fprintf(stderr, "cudaDeviceEnablePeerAccess(%d -> %d): %s\n", source, target, cudaGetErrorString(error));
            }
        }
    }
}

void runBenchmarkStencil(int dim, int iters, int workers, bool useDirect, FILE* csv_file) {
    if (workers < 1) workers = 1;
    if (dim % workers != 0) {
        fprintf(stderr, "DIM (%d) должен быть кратен количеству GPU (%d)\n", dim, workers);
        exit(EXIT_FAILURE);
    }

    if (useDirect && workers > 1) {
        enableP2P(workers);
    }

    int local_h = dim / workers;
    size_t row_bytes = dim * sizeof(float);
    size_t local_bytes = row_bytes * (local_h + 2);

    float* h_grid = (float*)calloc(dim * dim, sizeof(float));
    if (!h_grid) { fputs("OOM\n", stderr); exit(EXIT_FAILURE); }

    float** d_in = (float**)calloc((size_t)workers * 2, sizeof(float*));
    float** d_out = (float**)calloc((size_t)workers * 2, sizeof(float*));
    if (!d_in || !d_out) { fputs("OOM\n", stderr); exit(EXIT_FAILURE); }

    for (int device = 0; device < workers; device++) {
        CUDA_CHECK(cudaSetDevice(device));
        for (int buf = 0; buf < 2; buf++) {
            CUDA_CHECK(cudaMalloc(&d_in[device * 2 + buf], local_bytes));
            CUDA_CHECK(cudaMalloc(&d_out[device * 2 + buf], local_bytes));
            CUDA_CHECK(cudaMemset(d_in[device * 2 + buf], 0, local_bytes));
            CUDA_CHECK(cudaMemset(d_out[device * 2 + buf], 0, local_bytes));
        }
    }

    for (int i = 0; i < dim * dim; i++) {
        int x = i % dim;
        int y = i / dim;
        if ((x > 600) && (x < 1200) && (y > 620) && (y < 1202)) {
            h_grid[i] = MAX_TEMP;
        }
    }
    h_grid[dim*200+200] = (MAX_TEMP + MIN_TEMP) / 2;
    h_grid[dim*1400+200] = MIN_TEMP;

    for (int device = 0; device < workers; device++) {
        CUDA_CHECK(cudaSetDevice(device));
        int src_row = device * local_h - 1;
        int dst_row = 1;
        int rows_to_copy = local_h + 2;
        if (device == 0) { src_row = 0; dst_row = 0; rows_to_copy = local_h + 1; }
        if (device == workers - 1) { rows_to_copy = local_h + 1; }
        if (workers == 1) { src_row = 0; dst_row = 0; rows_to_copy = local_h; }

        CUDA_CHECK(cudaMemcpy2D(d_in[device * 2] + dst_row * dim, row_bytes,
                                h_grid + src_row * dim, row_bytes,
                                row_bytes, rows_to_copy, cudaMemcpyHostToDevice));
    }

    const double start = omp_get_wtime();
    int cur = 0;

    #pragma omp parallel num_threads(workers)
    {
        int device = omp_get_thread_num();
        CUDA_CHECK(cudaSetDevice(device));
        
        float* my_in[2]  = {d_in[device * 2], d_in[device * 2 + 1]};
        float* my_out[2] = {d_out[device * 2], d_out[device * 2 + 1]};

        for (int it = 0; it < iters; it++) {
            int count = local_h;
            int grid_x = (dim + 15) / 16;
            int grid_y = (count + 15) / 16;

            // 1. Вычисление
            dim3 blocks(grid_x, grid_y);
            dim3 threads(16, 16);
            blend_kernel<<<blocks, threads>>>(my_out[cur ^ 1], my_in[cur], dim, local_h);
            CUDA_CHECK(cudaDeviceSynchronize());

            // 2. Обмен данными
            if (workers == 1) {
                // ОДИН GPU: просто копируем результат из out в in для следующей итерации
                CUDA_CHECK(cudaMemcpy2D(my_in[cur ^ 1] + dim, row_bytes,
                                        my_out[cur ^ 1] + dim, row_bytes,
                                        row_bytes, local_h, cudaMemcpyDeviceToDevice));
            } else if (useDirect) {
                // MULTI-GPU P2P: обмен гало-слоями с соседями
                if (device < workers - 1) {
                    CUDA_CHECK(cudaMemcpyPeer(d_in[(device + 1) * 2 + cur], device + 1,
                                              my_out[cur ^ 1] + local_h * dim, device, row_bytes));
                }
                if (device > 0) {
                    CUDA_CHECK(cudaMemcpyPeer(d_in[(device - 1) * 2 + cur] + (local_h + 1) * dim, device - 1,
                                              my_out[cur ^ 1] + dim, device, row_bytes));
                }
                CUDA_CHECK(cudaDeviceSynchronize());
                
                // + локальное копирование своих строк
                CUDA_CHECK(cudaMemcpy2D(my_in[cur ^ 1] + dim, row_bytes,
                                        my_out[cur ^ 1] + dim, row_bytes,
                                        row_bytes, local_h, cudaMemcpyDeviceToDevice));
            } else {
                // MULTI-GPU Host Copy
                int first = device * local_h;
                CUDA_CHECK(cudaMemcpy(h_grid + first * dim, my_out[cur ^ 1] + dim,
                                      count * row_bytes, cudaMemcpyDeviceToHost));
            }

            #pragma omp barrier

            if (workers > 1 && !useDirect) {
                int src_row = device * local_h - 1;
                int dst_row = 1;
                int rows_to_copy = local_h + 2;
                if (device == 0) { src_row = 0; dst_row = 0; rows_to_copy = local_h + 1; }
                if (device == workers - 1) { rows_to_copy = local_h + 1; }

                CUDA_CHECK(cudaMemcpy2D(my_in[cur ^ 1] + dst_row * dim, row_bytes,
                                        h_grid + src_row * dim, row_bytes,
                                        row_bytes, rows_to_copy, cudaMemcpyHostToDevice));
            }

            #pragma omp barrier
            #pragma omp single
            { cur ^= 1; }
            #pragma omp barrier
        }
    }

    // 3. ФОРСИРУЕМ ЧТЕНИЕ РЕЗУЛЬТАТА. 
    // Это предотвращает Dead Code Elimination компилятором.
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMemcpy(h_grid, d_in[0], dim * sizeof(float), cudaMemcpyDeviceToHost));

    const double avg_ms = (omp_get_wtime() - start) * 1000.0 / iters;
    const long long cells = (long long)dim * dim * iters;
    const double mcells_s = (double)cells / (avg_ms * 1e-3) / 1e6;
    const char* variant = (useDirect || workers == 1) ? "stencil_direct" : "stencil_host";

    printf("%-18s %7d %5d %12.3f %14.3f\n", variant, dim, workers, avg_ms, mcells_s);
    
    if (csv_file) {
        fprintf(csv_file, "%s,%d,%d,%d,%.3f,%.3f\n", variant, workers, dim, iters, avg_ms, mcells_s);
        fflush(csv_file);
    }

    for (int device = 0; device < workers; device++) {
        CUDA_CHECK(cudaSetDevice(device));
        for (int buf = 0; buf < 2; buf++) {
            CUDA_CHECK(cudaFree(d_in[device * 2 + buf]));
            CUDA_CHECK(cudaFree(d_out[device * 2 + buf]));
        }
    }
    free(d_in); free(d_out); free(h_grid);
}

int main() {
    omp_set_dynamic(0);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 1) {
        fprintf(stderr, "CUDA GPU не найден\n");
        return EXIT_FAILURE;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s  (SM %d.%d, %zu MB)\n\n", prop.name, prop.major, prop.minor, prop.totalGlobalMem >> 20);

    FILE* csv_file = fopen("benchmark_results.csv", "w");
    if (csv_file) {
        fprintf(csv_file, "variant,gpus,dim,steps,avg_ms,mcells_s\n");
    }

    const int iters = STEPS;
    int max_workers = deviceCount;
    if (max_workers > omp_get_max_threads()) max_workers = omp_get_max_threads();

    const bool directAvailable = p2pAvailable(max_workers);

    printf("%-18s %7s %5s %12s %14s\n", "VARIANT", "DIM", "GPUs", "avg_ms", "MCells/s");
    printf("────────────────────────────────────────────────────────────\n");

    for (int workers = 1; workers <= max_workers; workers++) {
        printf("\n═══ Stencil Multi-GPU — %d GPU(s) ═══\n", workers);
        
        // Тестируем Host Copy (для 1 GPU это будет искусственно замедленный вариант)
        if (workers > 1) {
            runBenchmarkStencil(DIM, iters, workers, false, csv_file);
        }
        
        // Тестируем "Direct" (для 1 GPU это локальный DeviceToDevice, самый честный вариант)
        runBenchmarkStencil(DIM, iters, workers, true, csv_file);
    }

    if (csv_file) {
        fclose(csv_file);
        printf("\nРезультаты сохранены в benchmark_results.csv\n");
    }

    return 0;
}