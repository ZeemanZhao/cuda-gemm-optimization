#pragma once
#include <cuda_runtime.h>

// ─── Kernel 2: Shared Memory Tiling ──────────────────────────────────────────
//
// Classic "blocked matmul": decompose K into tiles of width TILE.
// Each block collaboratively loads a TILE×TILE sub-tile of A and B into
// __shared__ memory, then each thread accumulates from the fast on-chip SRAM.
//
// Memory access pattern:
//   Global → shared:  coalesced 128-byte transactions (TILE=32 → perfect)
//   Shared → register: bank-conflict-free when TILE is a multiple of 32
//
// Arithmetic intensity: TILE/2  (each element loaded once, used TILE times)
// Expected: ~20–40% of cuBLAS for TILE=32.
//
// Template param TILE controls tile size (32 is optimal for sm_89).

template <int TILE>
__global__ void sgemm_tiling(int M, int N, int K,
                              float alpha,
                              const float* __restrict__ A,   // [M, K]
                              const float* __restrict__ B,   // [K, N]
                              float beta,
                              float* __restrict__ C)         // [M, N]
{
    // On-chip tile buffers (TILE×TILE floats = 4 KB each for TILE=32)
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    // This thread's output coordinates
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.f;

    // Slide a TILE-wide window along K
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        // ── Load tile of A: row i, col t*TILE+threadIdx.x ──
        int aCol = t * TILE + threadIdx.x;
        As[threadIdx.y][threadIdx.x] = (row < M && aCol < K)
                                        ? A[row * K + aCol] : 0.f;

        // ── Load tile of B: row t*TILE+threadIdx.y, col j ──
        int bRow = t * TILE + threadIdx.y;
        Bs[threadIdx.y][threadIdx.x] = (bRow < K && col < N)
                                        ? B[bRow * N + col] : 0.f;

        __syncthreads();    // tiles fully loaded

        // ── Compute partial dot product from shared memory ──
        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();    // done reading; safe to overwrite next iter
    }

    if (row < M && col < N)
        C[row * N + col] = alpha * acc + beta * C[row * N + col];
}

// Launch wrapper
inline void launch_sgemm_tiling(int M, int N, int K,
                                  float alpha, const float* A, const float* B,
                                  float beta,  float* C)
{
    constexpr int TILE = 32;
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    sgemm_tiling<TILE><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
