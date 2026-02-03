#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

__global__ void matrixMultiplyTiled(const float *A, const float *B, float *C, int N) {
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Pvalue = 0.0f;

    // Number of tiles needed to cover the full N dimension
    int numTiles = (N + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int m = 0; m < numTiles; ++m) {
        // Load tile from A into shared memory
        int A_col = m * TILE_WIDTH + tx;
        if (Row < N && A_col < N) {
            ds_A[ty][tx] = A[Row * N + A_col];
        } else {
            ds_A[ty][tx] = 0.0f;
        }

        // Load tile from B into shared memory
        int B_row = m * TILE_WIDTH + ty;
        if (Col < N && B_row < N) {
            ds_B[ty][tx] = B[B_row * N + Col];
        } else {
            ds_B[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Multiply the two tiles
        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += ds_A[ty][k] * ds_B[k][tx];
        }

        __syncthreads();
    }

    // Store the result
    if (Row < N && Col < N) {
        C[Row * N + Col] = Pvalue;
    }
}

static void checkCuda(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    size_t size = (size_t)N * (size_t)N * sizeof(float);

    // Host allocations
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Host malloc failed\n");
        return 1;
    }

    // Initialize inputs
    srand(0);
    for (int i = 0; i < N * N; i++) {
        h_A[i] = (float)(rand() % 100) / 100.0f;
        h_B[i] = (float)(rand() % 100) / 100.0f;
    }

    // Device allocations
    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    checkCuda(cudaMalloc((void **)&d_A, size), "cudaMalloc d_A");
    checkCuda(cudaMalloc((void **)&d_B, size), "cudaMalloc d_B");
    checkCuda(cudaMalloc((void **)&d_C, size), "cudaMalloc d_C");

    // Copy inputs to device
    checkCuda(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice), "H2D copy A");
    checkCuda(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice), "H2D copy B");

    // Launch config: TILE_WIDTH x TILE_WIDTH threads
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((N + TILE_WIDTH - 1) / TILE_WIDTH,
              (N + TILE_WIDTH - 1) / TILE_WIDTH);

    // Time the kernel using CUDA events
    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "cudaEventCreate start");
    checkCuda(cudaEventCreate(&stop), "cudaEventCreate stop");

    checkCuda(cudaEventRecord(start), "cudaEventRecord start");
    matrixMultiplyTiled<<<grid, block>>>(d_A, d_B, d_C, N);
    checkCuda(cudaGetLastError(), "kernel launch");
    checkCuda(cudaEventRecord(stop), "cudaEventRecord stop");
    checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop");

    float ms = 0.0f;
    checkCuda(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");

    // Copy result back (not included in kernel time above)
    checkCuda(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost), "D2H copy C");

    printf("Tiled CUDA kernel time (N=%d): %.3f ms\n", N, ms);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
