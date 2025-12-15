#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define BSIZE2D 32
// Macro utilitario para captura de errores en llamadas a la API de CUDA
#define CHECK_CUDA(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "Error: " << __FILE__ << ":" << __LINE__ << ", " \
                  << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
}

// KERNEL: Acceso Indirecto a Memoria (Pattern: Gather)
__global__ void mikernel(float* A, const int* B, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) {
        int target_idx = B[tid];
        A[target_idx] = A[target_idx] + (float)tid;
    }
}

__global__ void kernel_matmul(int n, float *a, float *b, float *c){
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    float sum = 0.0f;
    for(int k=0; k<n; ++k){
        sum += a[ty*n + k] * b[k*n + tx];
    }
    c[ty*n + tx] = sum;
}

int main(int argc, char **argv){
    printf("GPU MATMUL\n");
    if(argc != 2){ fprintf(stderr, "run as ./prog n\n"); exit(EXIT_FAILURE); }
    int n = atoi(argv[1]); float msecs = 0.0f;

    // (1) creando matrices en host
    float *a = new float[n*n]; float *b = new float[n*n]; float *c = new float[n*n];
    float *cgold = new float[n*n];
    printf("initializing A and B......"); fflush(stdout);
    matrandom(n, a); matrandom(n, b);
    if(n < 64){
        printmat(n, a, "mat a"); printmat(n, b, "mat b");
    }
    printf("ok\n"); fflush(stdout);
    
    // (2) dejando matrices en device
    float *ad, *bd, *cd;
    cudaMalloc(&ad, sizeof(float)*n*n); cudaMalloc(&bd, sizeof(float)*n*n);
    cudaMalloc(&cd, sizeof(float)*n*n);
    cudaMemcpy(ad, a, sizeof(float)*n*n, cudaMemcpyHostToDevice);
    cudaMemcpy(bd, b, sizeof(float)*n*n, cudaMemcpyHostToDevice);
    cudaMemcpy(cd, c, sizeof(float)*n*n, cudaMemcpyHostToDevice);

    // (3) ejecutar matmul en GPU
    printf("computing C = A x B........"); fflush(stdout);
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    dim3 block(BSIZE2D, BSIZE2D, 1);
    dim3 grid((n+BSIZE2D-1)/BSIZE2D, (n+BSIZE2D-1)/BSIZE2D, 1);
    cudaEventRecord(start);
    kernel_matmul<<<grid, block>>>(n, ad, bd, cd);
    //kernel_matmulsm<<<grid, block>>>(n, ad, bd, cd);
    cudaDeviceSynchronize(); cudaEventRecord(stop);
    cudaEventSynchronize(stop); cudaEventElapsedTime(&msecs, start, stop);
    printf("ok: time: %f secs\n", msecs/1000.0f);

    // (4) copiar resultado a host
    printf("copying result to host....."); fflush(stdout);
    cudaMemcpy(c, cd, sizeof(float)*n*n, cudaMemcpyDeviceToHost);
    printf("ok\n"); fflush(stdout);
    if(n < 50){ printmat(n, c, "mat c"); }

    // (5) verificar resultado contra calculo en CPU
    /*printf("verifying result.........."); fflush(stdout);
    if(!verify(n, a, b, c, cgold)){
        fprintf(stderr, "error verifying result\n"); exit(EXIT_FAILURE);
    }
    */
    printf("ok\n");
    printf("done!\n");
    exit(EXIT_SUCCESS);
}
void matrandom(short int n, float * A){
    random_device rd;
    mt19937 generate(rd());
    uniform_real_distribution<> dis(0.0, 1.0);
    for (int i = 0; i < n * n; i++) { // n*n por que la matriz la simula un arreglo
        A[i] = dis(generate);
    }
}

void multiplicacionITL_Optimized(float* A, float* B, short int m, short int n, short int k) {
    float *C = new float[m * n];

    // 1. TRANSPONER MATRIZ B
    // Para aprovechar mejor la cache de la cpu en cada hilo
    float *B_T = new float[k * n];
    
    // Paralelizamos la transposición también
    #pragma omp parallel for collapse(2)
    for(int x = 0; x < k; x++) {
        for(int y = 0; y < n; y++) {
            B_T[y * k + x] = B[x * n + y];
        }
    }

    // 2. Calculo Matmul usando OMP para paralelizar con todos los nucleos
    #pragma omp parallel for
    for (int i = 0 ; i < m; i++){
        for (int j = 0; j < n; j++){
            
            float sum = 0;
            float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            float s4 = 0, s5 = 0, s6 = 0, s7 = 0;

            int l = 0;
            // Ahora B_T se accede como B_T[j * k + l], que es secuencial igual que A.
            for (; l <= k - 8; l += 8){
                // A accede fila i, B_T accede fila j (que es la columna j de B original)
                s0 += A[i * k + l]     * B_T[j * k + l];
                s1 += A[i * k + l + 1] * B_T[j * k + l + 1];
                s2 += A[i * k + l + 2] * B_T[j * k + l + 2];
                s3 += A[i * k + l + 3] * B_T[j * k + l + 3];
                s4 += A[i * k + l + 4] * B_T[j * k + l + 4];
                s5 += A[i * k + l + 5] * B_T[j * k + l + 5];
                s6 += A[i * k + l + 6] * B_T[j * k + l + 6];
                s7 += A[i * k + l + 7] * B_T[j * k + l + 7];
            }
            
            sum = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7;

            // Limpieza del resto
            for (; l < k; l++) {
                sum += A[i * k + l] * B_T[j * k + l];
            }
            
            C[i * n + j] = sum;
        }
    }

    cout << "(ITL OPTIMIZED) Matriz calculada" << endl;
    
    delete[] B_T; 
    delete[] C;
}

// Funcion para imprimir matrices visualmente ordenadas :D
void printmat(float* C, short int m, short int n){
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++){
            cout << C[i * n + j] << "  ";
        }
        cout << '\n';
    }
}

