#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

// ════════════════════════════════════════════════════════════════════
// Параметры задачи (как в исходной программе)
// ════════════════════════════════════════════════════════════════════
#define MAX_TEMP 1.0f
#define MIN_TEMP 0.0001f
#define SPEED    0.25f

#define TX 16
#define TY 16
#define BS (TX * TY)

// blend_kernel: 3 сложения + 1 умножение + 1 вычитание + 1 умножение + 1 сложение
#define FLOPS_PER_CELL 7

// ════════════════════════════════════════════════════════════════════
// Проверка ошибок
// ════════════════════════════════════════════════════════════════════
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t _e = (call);                                              \
        if (_e != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(_e));              \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// ════════════════════════════════════════════════════════════════════
// Текстура над pitch2D-буфером. Адресация clamp — как в исходной
// программе, где она получалась неявно при нормализованных = false.
// ════════════════════════════════════════════════════════════════════
static cudaTextureObject_t createTexture2D(float* devPtr, int width, int height) {
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = devPtr;
    resDesc.res.pitch2D.desc = cudaCreateChannelDesc<float>();
    resDesc.res.pitch2D.width = width;
    resDesc.res.pitch2D.height = height;
    resDesc.res.pitch2D.pitchInBytes = (size_t)width * sizeof(float);

    cudaTextureDesc texDesc = {};
    texDesc.readMode = cudaReadModeElementType;
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;

    cudaTextureObject_t tex;
    CUDA_CHECK(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));
    return tex;
}

// ════════════════════════════════════════════════════════════════════
// Ядра
// ════════════════════════════════════════════════════════════════════

// Перезаписывает ячейки-источники (cptr != 0). Работает по всем локальным
// строкам, включая halo: источники в halo-строках тоже должны быть применены.
__global__ void copy_const_kernel(float* iptr, const float* cptr, int N, int rows) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= N || y >= rows) return;

    int offset = x + y * N;
    if (cptr[offset] != 0) iptr[offset] = cptr[offset];
}

// Один шаг явной схемы теплопроводности для локальных строк [rowBegin, rowEnd).
// Выход за границу локального массива обрабатывается clamp-ом текстуры:
// для крайних GPU это и есть граница расчётной области, для внутренних
// соседние строки берутся из halo.
__global__ void blend_kernel(float* outSrc, cudaTextureObject_t texIn,
                             int N, int rowBegin, int rowEnd) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = rowBegin + threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= N || y >= rowEnd) return;

    float v     = tex2D<float>(texIn, x, y);
    float top   = tex2D<float>(texIn, x, y - 1);
    float bot   = tex2D<float>(texIn, x, y + 1);
    float left  = tex2D<float>(texIn, x - 1, y);
    float right = tex2D<float>(texIn, x + 1, y);

    outSrc[x + (size_t)y * N] = v + SPEED * (top + bot + left + right - 4 * v);
}

// ════════════════════════════════════════════════════════════════════
// Начальные условия (для N = 1024 совпадают с исходной программой,
// для других N координаты масштабируются)
// ════════════════════════════════════════════════════════════════════
static void initTemp(float* temp, int N) {
    const double s = N / 1024.0;
    auto sc = [&](int v) { return (int)(v * s); };

    memset(temp, 0, (size_t)N * N * sizeof(float));

    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            if (x > sc(300) && x < sc(600) && y > sc(310) && y < sc(601)) {
                temp[(size_t)y * N + x] = MAX_TEMP;
            }
        }
    }

    temp[(size_t)sc(100) * N + sc(100)] = (MAX_TEMP + MIN_TEMP) / 2;
    temp[(size_t)sc(700) * N + sc(100)] = MIN_TEMP;
    temp[(size_t)sc(300) * N + sc(300)] = MIN_TEMP;
    temp[(size_t)sc(200) * N + sc(700)] = MIN_TEMP;

    for (int y = sc(800); y < sc(900); y++) {
        for (int x = sc(400); x < sc(500); x++) {
            temp[(size_t)y * N + x] = MIN_TEMP;
        }
    }
}

// ════════════════════════════════════════════════════════════════════
// P2P
// ════════════════════════════════════════════════════════════════════
static bool p2pAvailable(int workers) {
    for (int source = 0; source < workers; source++) {
        for (int target = 0; target < workers; target++) {
            if (source == target) continue;
            int canAccess = 0;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccess, source, target));
            if (!canAccess) {
                fprintf(stderr, "GPU Direct недоступен: GPU %d -> GPU %d\n",
                        source, target);
                return false;
            }
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
                fprintf(stderr, "cudaDeviceEnablePeerAccess(%d -> %d): %s\n",
                        source, target, cudaGetErrorString(error));
                exit(EXIT_FAILURE);
            }
        }
    }
}

// ════════════════════════════════════════════════════════════════════
// Состояние одного GPU: горизонтальная полоса сетки + halo-строки
// ════════════════════════════════════════════════════════════════════
struct Dev {
    int first, rows;        // глобальный индекс первой своей строки и их число
    int top, bot;           // 1, если есть halo сверху / снизу
    int localRows;          // rows + top + bot
    float* buf[2];          // двойная буферизация (in/out)
    float* cst;             // источники тепла
    cudaTextureObject_t tex[2];
};

// ════════════════════════════════════════════════════════════════════
// Бенчмарк: сетка N×N, декомпозиция по строкам на `workers` GPU.
// useDirect = false: обмен halo через host; true: через cudaMemcpyPeer.
// Перед замером выполняется warmup шагов.
// ════════════════════════════════════════════════════════════════════
static void runBenchmarkHeat(int N, int steps, int warmup, int workers,
                             bool useDirect, FILE* csv_file)
{
    if (useDirect) enableP2P(workers);

    float* temp = (float*)malloc((size_t)N * N * sizeof(float));
    if (!temp) { fputs("OOM\n", stderr); exit(EXIT_FAILURE); }
    initTemp(temp, N);

    Dev* dev = (Dev*)calloc(workers, sizeof(Dev));
    float** sendUp   = (float**)calloc(workers, sizeof(float*));  // первая своя строка
    float** sendDown = (float**)calloc(workers, sizeof(float*));  // последняя своя строка
    if (!dev || !sendUp || !sendDown) { fputs("OOM\n", stderr); exit(EXIT_FAILURE); }

    // Выделение и загрузка делаются последовательно и не входят в замер.
    for (int d = 0; d < workers; d++) {
        Dev& me = dev[d];
        const int first = (int)((long long)N * d / workers);
        const int last  = (int)((long long)N * (d + 1) / workers);
        me.first = first;
        me.rows = last - first;
        me.top = (d > 0) ? 1 : 0;
        me.bot = (d < workers - 1) ? 1 : 0;
        me.localRows = me.rows + me.top + me.bot;

        const size_t bytes = (size_t)me.localRows * N * sizeof(float);
        const float* src = temp + (size_t)(first - me.top) * N;

        CUDA_CHECK(cudaSetDevice(d));
        for (int b = 0; b < 2; b++) {
            CUDA_CHECK(cudaMalloc(&me.buf[b], bytes));
            CUDA_CHECK(cudaMemset(me.buf[b], 0, bytes));
            me.tex[b] = createTexture2D(me.buf[b], N, me.localRows);
        }
        CUDA_CHECK(cudaMalloc(&me.cst, bytes));
        CUDA_CHECK(cudaMemcpy(me.cst,    src, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(me.buf[0], src, bytes, cudaMemcpyHostToDevice));

        sendUp[d]   = (float*)malloc((size_t)N * sizeof(float));
        sendDown[d] = (float*)malloc((size_t)N * sizeof(float));
        if (!sendUp[d] || !sendDown[d]) { fputs("OOM\n", stderr); exit(EXIT_FAILURE); }
    }

    double t0 = 0.0, t1 = 0.0;
    const size_t rowBytes = (size_t)N * sizeof(float);

    #pragma omp parallel num_threads(workers) shared(t0, t1)
    {
        const int d = omp_get_thread_num();
        CUDA_CHECK(cudaSetDevice(d));
        Dev& me = dev[d];
        int cur = 0;

        const dim3 threads(TX, TY);
        const dim3 gridC((N + TX - 1) / TX, (me.localRows + TY - 1) / TY);
        const dim3 gridB((N + TX - 1) / TX, (me.rows + TY - 1) / TY);

        for (int it = 0; it < warmup + steps; it++) {
            // Конец прогрева: все GPU синхронизируются, затем стартует таймер.
            if (it == warmup) {
                CUDA_CHECK(cudaDeviceSynchronize());
                #pragma omp barrier
                #pragma omp single
                t0 = omp_get_wtime();
            }

            copy_const_kernel<<<gridC, threads>>>(me.buf[cur], me.cst, N, me.localRows);
            blend_kernel<<<gridB, threads>>>(me.buf[cur ^ 1], me.tex[cur], N,
                                             me.top, me.top + me.rows);
            CUDA_CHECK(cudaGetLastError());

            // Обмен граничными строками результата (буфер cur^1) между соседями.
            if (workers > 1) {
                float* out = me.buf[cur ^ 1];
                const size_t firstOwn = (size_t)me.top * N;
                const size_t lastOwn  = (size_t)(me.top + me.rows - 1) * N;

                if (!useDirect) {
                    // Host copy: GPU -> host, барьер, host -> GPU.
                    if (d > 0)
                        CUDA_CHECK(cudaMemcpy(sendUp[d], out + firstOwn, rowBytes,
                                              cudaMemcpyDeviceToHost));
                    if (d < workers - 1)
                        CUDA_CHECK(cudaMemcpy(sendDown[d], out + lastOwn, rowBytes,
                                              cudaMemcpyDeviceToHost));
                    #pragma omp barrier
                    if (d > 0)
                        CUDA_CHECK(cudaMemcpy(out, sendDown[d - 1], rowBytes,
                                              cudaMemcpyHostToDevice));
                    if (d < workers - 1)
                        CUDA_CHECK(cudaMemcpy(out + (size_t)(me.localRows - 1) * N,
                                              sendUp[d + 1], rowBytes,
                                              cudaMemcpyHostToDevice));
                } else {
                    // GPU Direct: каждый GPU кладёт свои граничные строки
                    // прямо в halo соседей.
                    CUDA_CHECK(cudaDeviceSynchronize());
                    #pragma omp barrier
                    if (d > 0) {
                        Dev& up = dev[d - 1];
                        CUDA_CHECK(cudaMemcpyPeer(
                            up.buf[cur ^ 1] + (size_t)(up.localRows - 1) * N, d - 1,
                            out + firstOwn, d, rowBytes));
                    }
                    if (d < workers - 1) {
                        Dev& dn = dev[d + 1];
                        CUDA_CHECK(cudaMemcpyPeer(dn.buf[cur ^ 1], d + 1,
                                                  out + lastOwn, d, rowBytes));
                    }
                    CUDA_CHECK(cudaDeviceSynchronize());
                }
                #pragma omp barrier
            }

            cur ^= 1;
        }

        CUDA_CHECK(cudaDeviceSynchronize());
        #pragma omp barrier
        #pragma omp single
        t1 = omp_get_wtime();
    }

    const double avg_ms = (t1 - t0) * 1000.0 / steps;
    const double cells = (double)N * (double)N;
    const double gflops = cells * FLOPS_PER_CELL / (avg_ms * 1e-3) / 1e9;
    const char* variant = useDirect ? "multi_direct" : "multi_host";

    printf("%-14s %6d %5d %12.4f %14.3e %10.2f (%d GPU)\n",
           variant, N, BS, avg_ms, avg_ms / cells, gflops, workers);
    if (csv_file) {
        fprintf(csv_file, "%s,%d,float,%d,%d,%.4f,%.3e,%.2f\n",
                variant, workers, N, BS, avg_ms, avg_ms / cells, gflops);
        fflush(csv_file);
    }

    for (int d = 0; d < workers; d++) {
        CUDA_CHECK(cudaSetDevice(d));
        for (int b = 0; b < 2; b++) {
            CUDA_CHECK(cudaDestroyTextureObject(dev[d].tex[b]));
            CUDA_CHECK(cudaFree(dev[d].buf[b]));
        }
        CUDA_CHECK(cudaFree(dev[d].cst));
        free(sendUp[d]);
        free(sendDown[d]);
    }
    free(dev); free(sendUp); free(sendDown); free(temp);
}

// ════════════════════════════════════════════════════════════════════
// main
// ════════════════════════════════════════════════════════════════════
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
    printf("GPU: %s  (SM %d.%d, %zu MB)\n\n",
           prop.name, prop.major, prop.minor, prop.totalGlobalMem >> 20);

    // Принудительная инициализация контекстов всех GPU до замеров.
    for (int d = 0; d < deviceCount; d++) {
        CUDA_CHECK(cudaSetDevice(d));
        CUDA_CHECK(cudaFree(0));
    }

    FILE* csv_file = fopen("benchmark_results.csv", "w");
    if (csv_file) {
        fprintf(csv_file, "variant,gpus,type,N,BS,avg_ms,ms_per_cell,gflops\n");
    } else {
        fprintf(stderr, "Warning: Could not open benchmark_results.csv for writing\n");
    }

    const int N_list[] = {512, 1024, 2048, 4096, 8192};
    const int n_count  = sizeof(N_list) / sizeof(N_list[0]);
    const int steps    = 100;   // замеряемые шаги
    const int warmup   = 10;    // прогрев, в замер не входит

    int max_workers = deviceCount;
    if (max_workers > omp_get_max_threads()) {
        max_workers = omp_get_max_threads();
    }

    const bool directAvailable = p2pAvailable(max_workers);
    if (!directAvailable && max_workers > 1) {
        printf("GPU Direct пропущен: выбранные GPU не поддерживают P2P.\n");
    }

    for (int workers = 1; workers <= max_workers; workers++) {
        printf("\n═══ Heat 2D (float, texture, OpenMP) — %d GPU(s), %d шагов, прогрев %d ═══\n",
               workers, steps, warmup);
        printf("%-14s %6s %5s %12s %14s %10s\n",
               "VARIANT", "N", "BS", "avg_ms", "ms/cell", "GFLOP/s");
        printf("────────────────────────────────────────────────────────────────────\n");

        for (int ni = 0; ni < n_count; ni++) {
            const int N = N_list[ni];
            runBenchmarkHeat(N, steps, warmup, workers, false, csv_file);
            if (workers == 1 || directAvailable) {
                runBenchmarkHeat(N, steps, warmup, workers, true, csv_file);
            }
            puts("");
        }
    }

    if (csv_file) fclose(csv_file);
    return 0;
}
