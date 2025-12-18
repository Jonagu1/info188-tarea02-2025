all:
	nvcc main.cu -o prog -arch=sm_50 -allow-unsupported-compiler -Xcompiler -fopenmp -O3

clean:
	rm -f prog
