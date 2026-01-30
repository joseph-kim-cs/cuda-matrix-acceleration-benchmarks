#include <stdio.h>
#include <stdlib.h>
#include <time.h>
// all of this was already given

void matrixMultiplyCPU(float *A, float *B, float *C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main(int argc, char **argv) {
    // inputting matrix size N or default 1024
    int N = (argc > 1) ? atoi(argv[1]) : 1024;

    size_t size = N * N * sizeof(float);

    // allocating memory for matrices
    float *A = (float *)malloc(size);
    float *B = (float *)malloc(size);
    float *C = (float *)malloc(size);

    if (!A || !B || !C) {
        printf("Memory allocation failed\n");
        return 1;
    }

    // initializing matrices A and B with random values
    srand(0); 
    for (int i = 0; i < N * N; i++) {
        A[i] = (float)(rand() % 100) / 100.0f; // float values between 0.0 and 0.99
        B[i] = (float)(rand() % 100) / 100.0f;
    }

    // timing the matrix multiplication
    clock_t start = clock();
    matrixMultiplyCPU(A, B, C, N);
    clock_t end = clock();

    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("CPU execution time (N=%d): %.6f seconds\n", N, elapsed);
    // 512: 0.208000 seconds
    // 1024: 8.627000 seconds
    // 2048: 73.722000 seconds

    free(A); free(B); free(C);

    return 0;
}