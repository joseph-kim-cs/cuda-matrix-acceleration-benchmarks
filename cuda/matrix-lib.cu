#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

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

    for (int m = 0; m < (N + TILE_WIDTH - 1) / TILE_WIDTH; ++m) {

        if (Row < N && (m * TILE_WIDTH + tx) < N)
            ds_A[ty][tx] = A[Row * N + m * TILE_WIDTH + tx];
        else
            ds_A[ty][tx] = 0.0f;

        if (Col < N && (m * TILE_WIDTH + ty) < N)
            ds_B[ty][tx] = B[(m * TILE_WIDTH + ty) * N + Col];
        else
            ds_B[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
            Pvalue += ds_A[ty][k] * ds_B[k][tx];

        __syncthreads();
    }

    if (Row < N && Col < N)
        C[Row * N + Col] = Pvalue;
}

// exposed C interface for python
extern "C" void gpu_matrix_multiply(float *h_A, float *h_B, float *h_C, int N) {
    size_t size = N * N * sizeof(float);

    float *d_A, *d_B, *d_C;

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid(
        (N + TILE_WIDTH - 1) / TILE_WIDTH,
        (N + TILE_WIDTH - 1) / TILE_WIDTH
    );

    matrixMultiplyTiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}


// ADDING MORE LIBRARY FUNCTIONS FOR CONVOLUTION 

static inline void cudaCheck(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

// Naive 2D convolution: output[y, x] = sum_{j,i} input[y+j, x+i] * kernel[j,i]
__global__ void conv2d_same_kernel(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    float* __restrict__ output,
    int N,
    int K
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; //x will be column
    int y = blockIdx.y * blockDim.y + threadIdx.y; //y will be row

    if (x >= N || y >= N) return;

    int r = K / 2; // kernel radius
    float sum = 0.0f;

    for (int ky = 0; ky < K; ky++) {
        int in_y = y + (ky - r);
        if (in_y < 0 || in_y >= N) continue;

        for (int kx = 0; kx < K; kx++) {
            int in_x = x + (kx - r);
            if (in_x < 0 || in_x >= N) continue;

            float in_val = input[in_y * N + in_x];
            float k_val  = kernel[ky * K + kx];
            sum += in_val * k_val;
        }
    }

    output[y * N + x] = sum;
}

// input:  N*N float32
// kernel: K*K float32
// output: N*N float32
extern "C" void gpu_convolution_same(
    const float* input,
    const float* kernel,
    float* output,
    int N,
    int K
) {
    if (N <= 0 || K <= 0 || (K % 2) == 0) {
        fprintf(stderr, "gpu_convolution_same: N must be > 0 and K must be odd and > 0\n");
        exit(1);
    }

    size_t bytes_img = (size_t)N * (size_t)N * sizeof(float);
    size_t bytes_k   = (size_t)K * (size_t)K * sizeof(float);

    float *d_in = nullptr, *d_k = nullptr, *d_out = nullptr;

    cudaCheck(cudaMalloc((void**)&d_in, bytes_img), "cudaMalloc d_in");
    cudaCheck(cudaMalloc((void**)&d_k,  bytes_k),   "cudaMalloc d_k");
    cudaCheck(cudaMalloc((void**)&d_out, bytes_img),"cudaMalloc d_out");

    cudaCheck(cudaMemcpy(d_in, input, bytes_img, cudaMemcpyHostToDevice), "H2D input");
    cudaCheck(cudaMemcpy(d_k,  kernel, bytes_k,  cudaMemcpyHostToDevice), "H2D kernel");

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x,
              (N + block.y - 1) / block.y);

    conv2d_same_kernel<<<grid, block>>>(d_in, d_k, d_out, N, K);
    cudaCheck(cudaGetLastError(), "conv2d kernel launch");
    cudaCheck(cudaDeviceSynchronize(), "conv2d kernel sync");

    cudaCheck(cudaMemcpy(output, d_out, bytes_img, cudaMemcpyDeviceToHost), "D2H output");

    cudaFree(d_in);
    cudaFree(d_k);
    cudaFree(d_out);
}
