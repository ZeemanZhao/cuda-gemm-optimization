#pragma once
#include <cuda_runtime.h>

// ─── Kernel v2: Register Tiling (scalar global loads) ─────────────────────────
//
// Compared to v1 (1 thread = 1 output, MIO pipe bound):
//   - Each thread now computes TM×TN = 8×8 = 64 outputs (was 1).
//   - 64 independent register accumulators (regC[8][8]) → big ILP boost.
//   - shared loads per output drop ~32× (was 64 → now 2 per output element).
//   - Block thread count drops 1024 → 256 (1 thread covers 64 outputs instead).
//
// Compared to v3 (this file's float4-vectorized cousin):
//   - This version uses scalar global loads (4 separate floats per thread).
//   - v3 fuses 4 floats into a single float4 (128-bit LDG) load instruction.
//   - Splitting v2 / v3 lets us measure the contribution of each optimization.
//
// Block tile  : BM × BN = 128 × 128  outputs per block
// K-step      : BK = 8                inner-dim depth per outer iter
// Thread tile : TM × TN = 8 × 8       outputs per thread
// Block size  : (BM/TM) × (BN/TN) = 16 × 16 = 256 threads
//
// Shared memory: (BM*BK + BK*BN) × 4 = (1024 + 1024) × 4 = 8 KB / block
// Registers/thread: TM + TN + TM*TN = 80 floats (fits, no spill at sm_89)

template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_v2_register_tile(int M, int N, int K,
                                        float alpha,
                                        const float* __restrict__ A,   // [M, K]
                                        const float* __restrict__ B,   // [K, N]
                                        float beta,
                                        float* __restrict__ C)         // [M, N]
{
    constexpr int THREADS = (BM / TM) * (BN / TN);   // 256

    __shared__ float As[BM][BK];   // 128 × 8
    __shared__ float Bs[BK][BN];   // 8 × 128

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // Which 8×8 output sub-tile does this thread own?
    const int threadRow = tid / (BN / TN);   // 0..15
    const int threadCol = tid % (BN / TN);   // 0..15

    const int cRowBase = blockIdx.y * BM + threadRow * TM;
    const int cColBase = blockIdx.x * BN + threadCol * TN;

    // ── Accumulators & register tiles ──
    float regC[TM][TN] = {};
    float regA[TM];
    float regB[TN];

    // ── Load pattern: split 1024-element tile across 256 threads (4 scalars/thread) ──
    constexpr int A_LOADS = (BM * BK) / THREADS;   // 4
    constexpr int B_LOADS = (BK * BN) / THREADS;   // 4

    // As[lrow][lcol]: aRow walks BK columns; lrow strides BM rows
    const int aRow    = tid / BK;             // tid / 8 → 0..31
    const int aCol    = tid % BK;             // tid % 8 → 0..7
    const int aStride = THREADS / BK;         // 256 / 8 = 32 rows per iteration

    // Bs[lrow][lcol]: bCol covers BN columns; lrow strides BK rows
    const int bRow    = tid / BN;             // tid / 128 → 0..1
    const int bCol    = tid % BN;             // tid % 128 → 0..127
    const int bStride = THREADS / BN;         // 256 / 128 = 2 rows per iteration

    // ── Outer loop over K ──
    for (int ko = 0; ko < K; ko += BK) {

        // ── Scalar load: A tile (BM × BK) ──
        #pragma unroll
        for (int i = 0; i < A_LOADS; ++i) {
            const int lrow = aRow + i * aStride;
            const int lcol = aCol;
            const int grow = blockIdx.y * BM + lrow;
            const int gcol = ko + lcol;
            As[lrow][lcol] = (grow < M && gcol < K)
                            ? A[grow * K + gcol]
                            : 0.f;
        }

        // ── Scalar load: B tile (BK × BN) ──
        #pragma unroll
        for (int i = 0; i < B_LOADS; ++i) {
            const int lrow = bRow + i * bStride;
            const int lcol = bCol;
            const int grow = ko + lrow;
            const int gcol = blockIdx.x * BN + lcol;
            Bs[lrow][lcol] = (grow < K && gcol < N)
                            ? B[grow * N + gcol]
                            : 0.f;
        }

        __syncthreads();

        // ── Inner BK loop: each ki step is a TM×TN outer product ──
        #pragma unroll
        for (int ki = 0; ki < BK; ++ki) {
            // Load TM A-values (one column of As at this ki) into registers
            #pragma unroll
            for (int m = 0; m < TM; ++m)
                regA[m] = As[threadRow * TM + m][ki];

            // Load TN B-values (one row of Bs at this ki) into registers
            #pragma unroll
            for (int n = 0; n < TN; ++n)
                regB[n] = Bs[ki][threadCol * TN + n];

            // 64 independent FMAs into the regC accumulator
            #pragma unroll
            for (int m = 0; m < TM; ++m)
                #pragma unroll
                for (int n = 0; n < TN; ++n)
                    regC[m][n] += regA[m] * regB[n];
        }

        __syncthreads();
    }

    // ── Scalar writeback (TM × TN scalars per thread) ──
    #pragma unroll
    for (int m = 0; m < TM; ++m) {
        const int grow = cRowBase + m;
        if (grow >= M) continue;
        #pragma unroll
        for (int n = 0; n < TN; ++n) {
            const int gcol = cColBase + n;
            if (gcol < N)
                C[grow * N + gcol] = alpha * regC[m][n] +
                                     beta  * C[grow * N + gcol];
        }
    }
}

// Launch wrapper
inline void launch_sgemm_v2_register_tile(int M, int N, int K,
                                            float alpha, const float* A, const float* B,
                                            float beta,  float* C)
{
    constexpr int BM = 128, BN = 128, BK = 8;
    constexpr int TM = 8,   TN = 8;
    constexpr int BLOCK_X = BN / TN;   // 16
    constexpr int BLOCK_Y = BM / TM;   // 16
    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_v2_register_tile<BM, BN, BK, TM, TN><<<grid, block>>>(
        M, N, K, alpha, A, B, beta, C);
}
