// g++ -fopenmp matriz_ilp.cpp -o prog

#include <iostream>
#include <chrono>
#include <random>
#include <omp.h>
#include <vector>

using namespace std;

// Declaración de funciones
void multiplicacionDefault(float* A, float* B, short int m, short int n, short int k); //Sin ILP
void multiplicacionITL(float* A, float* B, short int m, short int n, short int k);//Con ILP
void multiplicacion(bool alg, float* A, float* B, short int m, short int n, short int k);//funcion para ver si usar ILP o no según "alg"
void printMatriz(float* C, short int m, short int n);

int main(int argc, char** argv){
    // Forzar la ejecucion con 5 argumentos
    if (argc != 5){
        cout << "Ejecutar: ./prog <m> <n> <k> <algoritmo>" << endl;
        return EXIT_FAILURE;
    }

    // Generador de números aleatorios
    random_device rd;
    mt19937 generate(rd());
    uniform_real_distribution<> dis(0.0, 1.0);

    // Lectura de dimensiones y alg
    short int m = atoi(argv[1]);
    short int n = atoi(argv[2]);
    short int k = atoi(argv[3]);
    bool alg = atoi(argv[4]); // Si es alg!=0 usa el algoritmo optimizado

    // Declarar matrices
    float *A = new float[m * k];
    float *B = new float[k * n];

    // Inicializar matrices con valores aleatorios
    for (int i = 0; i < m * k; i++) { // m*n por que la matriz la simula un arreglo
        A[i] = dis(generate);
    }
    for (int i = 0; i < k * n; i++) {
        B[i] = dis(generate);
    }

    // Medir tiempo de ejecución
    double t1 = omp_get_wtime(); //toma el tiempo antes
    multiplicacion(alg, A, B, m, n, k);
    double t2 = omp_get_wtime();//toma el tiempo despues
    double tiempo = t2 - t1; //calcula diferencia

    printf("> tiempo de suma = %f secs\n", tiempo);

    delete[] A;
    delete[] B;

    return EXIT_SUCCESS;
}

// Selección del algoritmo
void multiplicacion(bool alg, float* A, float* B, short int m, short int n, short int k){
    if (alg) { // Si alg == 1, usa versión optimizada
        multiplicacionITL_Optimized(A, B, m, n, k);
        return;
    }
    multiplicacionDefault(A, B, m, n, k);
}

// Algoritmo clásico (triple for)
void multiplicacionDefault(float* A, float* B, short int m, short int n, short int k) {
    float *C = new float[m * n];

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++){
            float sum = 0;
            for (int l = 0; l < k; l++){
                sum += A[i * k + l] * B[l * n + j]; 
            }
            C[i * n + j] = sum;
        }
    }

    cout << "(DEF) Matriz" << endl;
    // printMatriz(C, m, n);

    delete[] C;
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
void printMatriz(float* C, short int m, short int n){
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++){
            cout << C[i * n + j] << "  ";
        }
        cout << '\n';
    }
}

