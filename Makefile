NVCC = nvcc
TARGET = heat_bench.o
SRC = heat_bench.cu
RES = benchmark_results.csv

# Список GPU, доступных программе. Можно переопределить:
# make run GPUS=0,1,2,3
GPUS ?= 0,1

.PHONY: all build run gpus plot clean-data clean

all: build

build: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) -O3 -arch=native -Xcompiler -fopenmp $(SRC) -o $(TARGET)

# Принудительный запуск benchmark на выбранных GPU (всегда пересчитывает).
run: $(TARGET)
	CUDA_VISIBLE_DEVICES=$(GPUS) ./$(TARGET)

# CSV строится, только если его ещё нет. Бинарник нужен лишь в этом случае
# (order-only зависимость), поэтому на клиенте без nvcc с готовым CSV
# ничего не собирается и не запускается.
$(RES): | $(TARGET)
	CUDA_VISIBLE_DEVICES=$(GPUS) ./$(TARGET)

# Построение графиков через uv.
plot: $(RES)
	uv run plot

gpus:
	nvidia-smi -L

# Удалить все изображения и CSV-результаты benchmark.
clean-data:
	rm -f *.png *.csv

# Полная очистка: бинарник, изображения и CSV.
clean: clean-data
	rm -f $(TARGET)
