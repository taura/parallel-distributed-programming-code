NVCXX := /usr/local/cuda/bin/nvcc

all : hello_gpu hello_gpu2

hello_gpu : hello_gpu.cu
	$(NVCXX) -o $@ $< 

hello_gpu2 : hello_gpu2.cu
	$(NVCXX) -o $@ $< 

clean :
	rm -f hello_gpu hello_gpu2
