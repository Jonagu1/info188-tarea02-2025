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

void matrandom(int n, float *A);
void printmat(int n, float* C, const char* name);
void cpu_matrix_mult_optimized(float* A, float* B, float* C, int n);

// --- KERNEL GPU BÁSICO ---
__global__ void kernel_matmul(int n, float *a, float *b, float *c){
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    // Chequeo de bordes
    if (tx < n && ty < n) {
        float sum = 0.0f;
        for(int k=0; k<n; ++k){
            sum += a[ty*n + k] * b[k*n + tx];
        }
        c[ty*n + tx] = sum;
    }
}

// --- KERNEL GPU CON MEMORIA COMPARTIDA ---
__global__ void kernel_matmul_shared(int n, float *a, float *b, float *c){
    __shared__ float As[BSIZE2D][BSIZE2D];
    __shared__ float Bs[BSIZE2D][BSIZE2D];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int col = blockIdx.x * BSIZE2D + tx;
    int row = blockIdx.y * BSIZE2D + ty;

    float sum = 0.0f;

    for (int m = 0; m < (n + BSIZE2D - 1) / BSIZE2D; m++) {

        //Cargar A
        if (row < n && (m * BSIZE2D + tx) < n)
            As[ty][tx] = a[row * n + m * BSIZE2D + tx];
        else
            As[ty][tx] = 0.0f;

        //Cargar B
        if (col < n && (m * BSIZE2D + ty) < n)
            Bs[ty][tx] = b[(m * BSIZE2D + ty) * n + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < BSIZE2D; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    //Guardar resultado
    if (row < n && col < n) {
        c[row * n + col] = sum;
    }
}


// --- MAIN ---
int main(int argc, char **argv){

    // 1. ARGUMENTOS
    if(argc != 4){
        fprintf(stderr, "Uso: ./prog <n> <nt> <alg>\n");
        fprintf(stderr, "   <n>   : Tamaño de matriz\n");
        fprintf(stderr, "   <nt>  : Hilos OpenMP\n");
        fprintf(stderr, "   <alg> : 1=CPU; 2=GPU; 3=GPU Shared\n");
        exit(EXIT_FAILURE);
    }

    int n = atoi(argv[1]);
    int nt = atoi(argv[2]);
    int alg = atoi(argv[3]);

    if (alg < 1 || alg > 3) {
            fprintf(stderr, "Algoritmo inválido\n");
            exit(EXIT_FAILURE);
    }

    // Configurar OpenMP
    omp_set_num_threads(nt);

    printf("[INFO] Config: N = %d, Hilos = %d, Modo = %s\n", n, nt, alg == 1 ? "CPU" : "GPU");

    // (1) creando matrices en host
    float *a = new float[n*n];
    float *b = new float[n*n];
    float *c = new float[n*n];

    printf("[INFO] Inicializando A y B......"); fflush(stdout);
    matrandom(n, a);
    matrandom(n, b);

    if(n < 16){
        printmat(n, a, "Matriz A");
        printmat(n, b, "Matriz B");
    }

    // -----------------------------------------------------------
    // LOGICA DE SELECCION DE ALGORITMO
    // -----------------------------------------------------------

    if (alg == 1) {
        // MODO CPU
        printf("\n[INFO] Ejecutando multiplicación en CPU (OpenMP)\n");
        double t1 = omp_get_wtime();

        cpu_matrix_mult_optimized(a, b, c, n);

        double t2 = omp_get_wtime();
        printf("[INFO] Tiempo: %f ms\n", (t2 - t1) * 1000.0f);

    } else {
        // MODO GPU
        printf("\n[INFO] Ejecutando multiplicación en GPU (CUDA)\n");

        // (2) dejando matrices en device
        float *ad, *bd, *cd;
        float msecs = 0.0f;

        CHECK_CUDA(cudaMalloc(&ad, sizeof(float)*n*n));
        CHECK_CUDA(cudaMalloc(&bd, sizeof(float)*n*n));
        CHECK_CUDA(cudaMalloc(&cd, sizeof(float)*n*n));

        CHECK_CUDA(cudaMemcpy(ad, a, sizeof(float)*n*n, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(bd, b, sizeof(float)*n*n, cudaMemcpyHostToDevice));

        // (3) ejecutar matmul en GPU
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        dim3 block(BSIZE2D, BSIZE2D, 1);
        dim3 grid((n+BSIZE2D-1)/BSIZE2D, (n+BSIZE2D-1)/BSIZE2D, 1);

        CHECK_CUDA(cudaEventRecord(start));

        if (alg == 2){
            kernel_matmul<<<grid, block>>>(n, ad, bd, cd);
        } else if (alg == 3){
            kernel_matmul_shared <<<grid, block>>> (n, ad, bd, cd);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&msecs, start, stop));
        CHECK_CUDA(cudaGetLastError()); // Verificar errores de kernel

        printf("[INFO] Tiempo: %f ms\n", msecs);

        // (4) copiar resultado a host
        printf("[INFO] Copiando resultado desde GPU al host...\n"); fflush(stdout);
        CHECK_CUDA(cudaMemcpy(c, cd, sizeof(float)*n*n, cudaMemcpyDeviceToHost));
        printf("[INFO] Resultado copiado\n");

        // Limpieza GPU
        cudaFree(ad); cudaFree(bd); cudaFree(cd);
    }

    // Imprimir resultado si es pequeño
    if(n < 16){
        printmat(n, c, "Matriz C (Resultado)");
    }

    // Limpieza Host
    delete[] a; delete[] b; delete[] c;

    printf("[INFO] Ejecución finalizada correctamente\n");
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
