#pragma once
#include <cuda_runtime.h>

// ─── Kernel 1: Naive SGEMM ────────────────────────────────────────────────────
//
// Each thread computes one output element C[row][col].
// Reads entire row of A and column of B from global memory — fully uncoalesced
// for A, coalesced for B (col-major access is bad here).
//
// Bottleneck: global memory latency. Each thread issues 2*K uncached loads.
// Expected performance: ~3–8% of cuBLAS on large matrices.
//
// C = alpha * A @ B + beta * C   (A: M×K, B: K×N, C: M×N, row-major)

__global__ void sgemm_naive(int M, int N, int K,
                             float alpha,
                             const float* __restrict__ A,   // [M, K]
                             const float* __restrict__ B,   // [K, N]
                             float beta,
                             float* __restrict__ C)         // [M, N]
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float acc = 0.f;
    for (int k = 0; k < K; ++k)
        acc += A[row * K + k] * B[k * N + col];

    C[row * N + col] = alpha * acc + beta * C[row * N + col];
}

// Launch wrapper
inline void launch_sgemm_naive(int M, int N, int K,
                                float alpha, const float* A, const float* B,
                                float beta,  float* C)
{
    // 32×32 = 1024 threads/block — one thread per output element
    constexpr int BLOCK = 32;
    dim3 block(BLOCK, BLOCK);
    dim3 grid((N + BLOCK - 1) / BLOCK, (M + BLOCK - 1) / BLOCK);
    sgemm_naive<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
