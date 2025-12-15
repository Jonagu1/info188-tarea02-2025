all:
	nvcc -O3 -arch=sm_50 -std=c++14 main.cu -o prog

clean:
	rm -f prog