#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <cuda_runtime.h>

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

int main(int argc, char** argv) {
    // 1. Parsing de argumentos y configuración del entorno
    if (argc != 3) {
        std::cerr << "Uso: " << argv[0] << " <n> <modo>" << std::endl;
        return 1;
    }
    int n = std::atoi(argv[1]);
    int mode = std::atoi(argv[2]);

    std::cout << "Configuracion: N=" << n << ", Modo=" << (mode == 0 ? "Ordenado" : "Aleatorio") << std::endl;

    // 2. Preparación de datos en Host
    // Se inicializa A con ruido y B como vector de permutación.
    std::vector<float> h_A(n);
    std::vector<int> h_B(n);
    
    std::mt19937 gen(1234);
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (int i = 0; i < n; ++i) h_A[i] = dis(gen);
    std::iota(h_B.begin(), h_B.end(), 0);

    if (mode == 1) {
        std::shuffle(h_B.begin(), h_B.end(), gen);
    }

    // 3. Gestión de Memoria en Device (Allocation & Transfer)
    float *d_A;
    int *d_B;
    CHECK_CUDA(cudaMalloc(&d_A, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, n * sizeof(int)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    // 4. Configuración y Ejecución del Kernel (Profiling)
    // Se utiliza eventos CUDA para medir estrictamente el tiempo de cómputo GPU
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    mikernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, n);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    CHECK_CUDA(cudaGetLastError());
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Tiempo de Kernel: " << milliseconds << " ms" << std::endl;

    // 5. Liberación de recursos
    cudaFree(d_A);
    cudaFree(d_B);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
