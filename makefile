all:
	nvcc main.cu -o prog -arch=sm_50 -Xcompiler -fopenmp -O3

clean:
	rm -f prog