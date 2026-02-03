#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

static void checkCuda(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

static void checkCublas(cublasStatus_t stat, const char *msg) {
    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS error: %s (status=%d)\n", msg, (int)stat);
        exit(1);
    }
}

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    size_t size = (size_t)N * (size_t)N * sizeof(float);

    // host allocations
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Host malloc failed\n");
        return 1;
    }

    // initialize matrices
    srand(0);
    for (int i = 0; i < N * N; i++) {
        h_A[i] = (float)(rand() % 100) / 100.0f;
        h_B[i] = (float)(rand() % 100) / 100.0f;
    }

    // device allocations
    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    checkCuda(cudaMalloc((void **)&d_A, size), "cudaMalloc d_A");
    checkCuda(cudaMalloc((void **)&d_B, size), "cudaMalloc d_B");
    checkCuda(cudaMalloc((void **)&d_C, size), "cudaMalloc d_C");

    checkCuda(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice), "H2D copy A");
    checkCuda(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice), "H2D copy B");

    cublasHandle_t handle;
    checkCublas(cublasCreate(&handle), "cublasCreate");

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // start the timer
    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "cudaEventCreate start");
    checkCuda(cudaEventCreate(&stop), "cudaEventCreate stop");

    checkCuda(cudaEventRecord(start), "cudaEventRecord start");

    checkCublas(
        cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    N, N, N,
                    &alpha,
                    d_B, N, 
                    d_A, N, 
                    &beta,
                    d_C, N),
        "cublasSgemm"
    );

    // stopping the timer
    checkCuda(cudaEventRecord(stop), "cudaEventRecord stop");
    checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop");

    float ms = 0.0f;
    checkCuda(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");

    checkCuda(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost), "D2H copy C");

    printf("cuBLAS SGEMM time (N=%d): %.3f ms\n", N, ms);

    // cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
