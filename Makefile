NVCC = nvcc
TARGET = heat_bench.o
SRC = heat_bench.cu

# Список GPU, доступных программе. Можно переопределить:
# make run GPUS=0,1,2,3
GPUS ?= 0,1

.PHONY: all build run gpus plot clean-data clean

all: build

build: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) -O3 -arch=native -Xcompiler -fopenmp $(SRC) -o $(TARGET)

# Запуск benchmark на выбранных GPU.
run: $(TARGET)
	CUDA_VISIBLE_DEVICES=$(GPUS) ./$(TARGET)

# Построение графиков и сводных CSV через uv.
plot: run
	uv run plot.py

gpus:
	nvidia-smi -L

# Удалить все изображения и CSV-результаты benchmark.
clean-data:
	rm -f *.png *.csv

# Полная очистка: бинарник, изображения и CSV.
clean: clean-data
	rm -f $(TARGET)
