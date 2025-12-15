#include <iostream>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

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

using namespace std;

// --- PROTOTIPOS ---
void matrandom(int n, float *A);
void printmat(int n, float* C, const char* name);
void cpu_matrix_mult_optimized(float* A, float* B, float* C, int n);

// --- KERNEL GPU ---
__global__ void kernel_matmul(int n, float *a, float *b, float *c){
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    // Chequeo de bordes
    if (tx < n && ty < n) {
        float sum = 0.0f; // Faltaba inicializar esto en tu codigo original
        for(int k=0; k<n; ++k){
            sum += a[ty*n + k] * b[k*n + tx];
        }
        c[ty*n + tx] = sum;
    }
}

// --- MAIN ---
int main(int argc, char **argv){
    
    // 1. ARGUMENTOS
    if(argc != 4){ 
        fprintf(stderr, "Uso: ./prog <n> <nt> <alg>\n"); 
        fprintf(stderr, "   <n>   : Tamano de matriz\n");
        fprintf(stderr, "   <nt>  : Hilos OpenMP\n");
        fprintf(stderr, "   <alg> : 0=CPU, 1=GPU\n");
        exit(EXIT_FAILURE); 
    }

    int n = atoi(argv[1]); 
    int nt = atoi(argv[2]);
    int alg = atoi(argv[3]);

    // Configurar OpenMP
    omp_set_num_threads(nt);

    printf("Config: N=%d, Threads=%d, Mode=%s\n", n, nt, alg == 0 ? "CPU" : "GPU");

    // (1) creando matrices en host
    float *a = new float[n*n]; 
    float *b = new float[n*n]; 
    float *c = new float[n*n];
    
    printf("initializing A and B......"); fflush(stdout);
    matrandom(n, a); 
    matrandom(n, b);
    
    if(n < 16){
        printmat(n, a, "Matriz A"); 
        printmat(n, b, "Matriz B");
    }
    printf("ok\n"); fflush(stdout);
    
    // -----------------------------------------------------------
    // LOGICA DE SELECCION DE ALGORITMO
    // -----------------------------------------------------------
    
    if (alg == 0) {
        // >>> MODO CPU <<<
        printf("Computing on CPU (OpenMP)........\n");
        double t1 = omp_get_wtime();
        
        cpu_matrix_mult_optimized(a, b, c, n);
        
        double t2 = omp_get_wtime();
        printf("ok: time: %f secs\n", t2 - t1);

    } else {
        // >>> MODO GPU <<<
        printf("Computing on GPU (CUDA)........\n");
        
        // (2) dejando matrices en device
        float *ad, *bd, *cd;
        float msecs = 0.0f;

        CHECK_CUDA(cudaMalloc(&ad, sizeof(float)*n*n)); 
        CHECK_CUDA(cudaMalloc(&bd, sizeof(float)*n*n));
        CHECK_CUDA(cudaMalloc(&cd, sizeof(float)*n*n));
        
        CHECK_CUDA(cudaMemcpy(ad, a, sizeof(float)*n*n, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(bd, b, sizeof(float)*n*n, cudaMemcpyHostToDevice));
        // No copiamos C a device porque es solo salida, pero si quieres limpiar basura:
        // cudaMemset(cd, 0, sizeof(float)*n*n);

        // (3) ejecutar matmul en GPU
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start)); 
        CHECK_CUDA(cudaEventCreate(&stop));
        
        dim3 block(BSIZE2D, BSIZE2D, 1);
        dim3 grid((n+BSIZE2D-1)/BSIZE2D, (n+BSIZE2D-1)/BSIZE2D, 1);
        
        CHECK_CUDA(cudaEventRecord(start));
        kernel_matmul<<<grid, block>>>(n, ad, bd, cd);
        CHECK_CUDA(cudaEventRecord(stop));
        
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&msecs, start, stop));
        CHECK_CUDA(cudaGetLastError()); // Verificar errores de kernel
        
        printf("ok: time: %f secs\n", msecs/1000.0f);

        // (4) copiar resultado a host
        printf("copying result to host....."); fflush(stdout);
        CHECK_CUDA(cudaMemcpy(c, cd, sizeof(float)*n*n, cudaMemcpyDeviceToHost));
        printf("ok\n");

        // Limpieza GPU
        cudaFree(ad); cudaFree(bd); cudaFree(cd);
    }

    // Imprimir resultado si es pequeño
    if(n < 16){ 
        printmat(n, c, "Matriz C (Resultado)"); 
    }

    // Limpieza Host
    delete[] a; delete[] b; delete[] c;
    
    printf("done!\n");
    return EXIT_SUCCESS;
}

// --- IMPLEMENTACIONES ---

void matrandom(int n, float * A){
    static std::mt19937 generate(1234); 
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (int i = 0; i < n * n; i++) {
        A[i] = (float)dis(generate);
    }
}

void printmat(int n, float* C, const char* name){
    cout << "\n[" << name << "]\n";
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++){
            printf("%5.2f ", C[i * n + j]);
        }
        cout << '\n';
    }
}

void cpu_matrix_mult_optimized(float* A, float* B, float* C_out , int n) {
    float *B_T = new float[n * n];
    
    // Transponer B
    #pragma omp parallel for collapse(2)
    for(int x = 0; x < n; x++) {
        for(int y = 0; y < n; y++) {
            B_T[y * n + x] = B[x * n + y];
        }
    }

    // Calculo Matmul
    #pragma omp parallel for
    for (int i = 0 ; i < n; i++){
        for (int j = 0; j < n; j++){
            
            float sum = 0;
            float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            float s4 = 0, s5 = 0, s6 = 0, s7 = 0;
            int l = 0;
            
            // Reemplacé 'k' (que no existía) por 'n'
            for (; l <= n - 8; l += 8){
                s0 += A[i * n + l]     * B_T[j * n + l];
                s1 += A[i * n + l + 1] * B_T[j * n + l + 1];
                s2 += A[i * n + l + 2] * B_T[j * n + l + 2];
                s3 += A[i * n + l + 3] * B_T[j * n + l + 3];
                s4 += A[i * n + l + 4] * B_T[j * n + l + 4];
                s5 += A[i * n + l + 5] * B_T[j * n + l + 5];
                s6 += A[i * n + l + 6] * B_T[j * n + l + 6];
                s7 += A[i * n + l + 7] * B_T[j * n + l + 7];
            }
            
            sum = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7;

            // Limpieza del resto
            for (; l < n; l++) {
                sum += A[i * n + l] * B_T[j * n + l];
            }
            
            C_out[i * n + j] = sum;
        }
    }
    delete[] B_T; 
}
