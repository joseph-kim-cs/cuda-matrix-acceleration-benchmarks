#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_WIDTH 16

__global__ void matrixMultiplyTiled(float *A, float *B, float *C, int N) {
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Pvalue = 0.0f;

    // loop over tiles
    for (int m = 0; m < (N + TILE_WIDTH - 1) / TILE_WIDTH; ++m) {

        // load tile of A into shared memory
        if (Row < N && (m * TILE_WIDTH + tx) < N)
            ds_A[ty][tx] = A[Row * N + m * TILE_WIDTH + tx];
        else
            ds_A[ty][tx] = 0.0f;

        // load tile of B into shared memory
        if (Col < N && (m * TILE_WIDTH + ty) < N)
            ds_B[ty][tx] = B[(m * TILE_WIDTH + ty) * N + Col];
        else
            ds_B[ty][tx] = 0.0f;

        __syncthreads();

        // compute partial dot product
        for (int k = 0; k < TILE_WIDTH; ++k)
            Pvalue += ds_A[ty][k] * ds_B[k][tx];

        __syncthreads();
    }

    // write result
    if (Row < N && Col < N)
        C[Row * N + Col] = Pvalue;
}
