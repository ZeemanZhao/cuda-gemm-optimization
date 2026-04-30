#pragma once
#include <cuda_runtime.h>

// ─── Kernel v5: Double Buffering (software pipelining) ────────────────────────
//
// Same algorithm as v3 (register tile + float4 vectorized loads), but the
// global→shared loads of the *next* K-tile run in parallel with the
// shared→register compute of the *current* K-tile. This hides global-load
// latency behind FMA work.
//
// How:
//   __shared__ float As[2][BM][BK];   // two buffers
//   __shared__ float Bs[2][BK][BN];
//
// Pipeline structure:
//   1. Initial load: tile 0 → As[0], Bs[0]   (single sync after)
//   2. Main loop, for k = 1 .. (K/BK - 1):
//        a. Prefetch tile k → As[load_idx], Bs[load_idx]
//        b. Compute tile k-1 from As[comp_idx], Bs[comp_idx]   (parallel)
//        c. __syncthreads() — wait for both to finish
//        d. Swap comp_idx ↔ load_idx
//   3. Final compute: tile (K/BK - 1) from As[comp_idx], Bs[comp_idx]
//
// Why it helps:
//   v3 issues all A/B loads before computing → LSU pipe burst, then idle
//   while FMA pipe runs. v5 interleaves: while FMA pipe is busy on tile k-1,
//   LSU pipe is fetching tile k. Both pipes stay closer to peak utilization.
//
// Cost: 2× shared memory (8 KB → 16 KB / block). Still under SM limit on Ada.
// Expected: 5–15% speedup over v3 on Ada (sm_89).

template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_v5_double_buffer(int M, int N, int K,
                                         float alpha,
                                         const float* __restrict__ A,
                                         const float* __restrict__ B,
                                         float beta,
                                         float* __restrict__ C)
{
    static_assert(BK % 4 == 0, "BK must be divisible by 4");
    static_assert(BN % 4 == 0, "BN must be divisible by 4");
    static_assert(TN % 4 == 0, "TN must be divisible by 4 (for vectorized writeback)");

    constexpr int THREADS = (BM / TM) * (BN / TN);   // 256

    // Two-stage shared buffers
    __shared__ float As[2][BM][BK];   // 2 × 128 × 8 = 8 KB
    __shared__ float Bs[2][BK][BN];   // 2 × 8 × 128 = 8 KB

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;

    const int threadRow = tid / (BN / TN);
    const int threadCol = tid % (BN / TN);

    const int cRowBase = blockIdx.y * BM + threadRow * TM;
    const int cColBase = blockIdx.x * BN + threadCol * TN;

    float regC[TM][TN] = {};
    float regA[TM];
    float regB[TN];

    constexpr int A_LOADS = (BM * BK) / (THREADS * 4);
    constexpr int B_LOADS = (BK * BN) / (THREADS * 4);

    const int aRow    = tid / (BK / 4);
    const int aCol4   = tid % (BK / 4);
    const int aStride = THREADS / (BK / 4);

    const int bRow    = tid / (BN / 4);
    const int bCol4   = tid % (BN / 4);
    const int bStride = THREADS / (BN / 4);

    // ── Initial load: tile 0 → buffer 0 ──
    {
        const int ko = 0;
        #pragma unroll
        for (int i = 0; i < A_LOADS; ++i) {
            const int lrow = aRow + i * aStride;
            const int lcol = aCol4 * 4;
            const int grow = blockIdx.y * BM + lrow;
            const int gcol = ko + lcol;
            float4 v = {};
            if (grow < M) {
                if (gcol + 3 < K) {
                    v = *reinterpret_cast<const float4*>(&A[grow * K + gcol]);
                } else {
                    v.x = (gcol     < K) ? A[grow * K + gcol    ] : 0.f;
                    v.y = (gcol + 1 < K) ? A[grow * K + gcol + 1] : 0.f;
                    v.z = (gcol + 2 < K) ? A[grow * K + gcol + 2] : 0.f;
                    v.w = (gcol + 3 < K) ? A[grow * K + gcol + 3] : 0.f;
                }
            }
            As[0][lrow][lcol    ] = v.x;
            As[0][lrow][lcol + 1] = v.y;
            As[0][lrow][lcol + 2] = v.z;
            As[0][lrow][lcol + 3] = v.w;
        }
        #pragma unroll
        for (int i = 0; i < B_LOADS; ++i) {
            const int lrow = bRow + i * bStride;
            const int lcol = bCol4 * 4;
            const int grow = ko + lrow;
            const int gcol = blockIdx.x * BN + lcol;
            float4 v = {};
            if (grow < K) {
                if (gcol + 3 < N) {
                    v = *reinterpret_cast<const float4*>(&B[grow * N + gcol]);
                } else {
                    v.x = (gcol     < N) ? B[grow * N + gcol    ] : 0.f;
                    v.y = (gcol + 1 < N) ? B[grow * N + gcol + 1] : 0.f;
                    v.z = (gcol + 2 < N) ? B[grow * N + gcol + 2] : 0.f;
                    v.w = (gcol + 3 < N) ? B[grow * N + gcol + 3] : 0.f;
                }
            }
            Bs[0][lrow][lcol    ] = v.x;
            Bs[0][lrow][lcol + 1] = v.y;
            Bs[0][lrow][lcol + 2] = v.z;
            Bs[0][lrow][lcol + 3] = v.w;
        }
    }
    __syncthreads();

    int comp_idx = 0;

    // ── Main loop: prefetch tile k while computing tile k-1 ──
    for (int ko = BK; ko < K; ko += BK) {
        const int load_idx = 1 - comp_idx;

        // Prefetch tile k → As[load_idx], Bs[load_idx]
        #pragma unroll
        for (int i = 0; i < A_LOADS; ++i) {
            const int lrow = aRow + i * aStride;
            const int lcol = aCol4 * 4;
            const int grow = blockIdx.y * BM + lrow;
            const int gcol = ko + lcol;
            float4 v = {};
            if (grow < M) {
                if (gcol + 3 < K) {
                    v = *reinterpret_cast<const float4*>(&A[grow * K + gcol]);
                } else {
                    v.x = (gcol     < K) ? A[grow * K + gcol    ] : 0.f;
                    v.y = (gcol + 1 < K) ? A[grow * K + gcol + 1] : 0.f;
                    v.z = (gcol + 2 < K) ? A[grow * K + gcol + 2] : 0.f;
                    v.w = (gcol + 3 < K) ? A[grow * K + gcol + 3] : 0.f;
                }
            }
            As[load_idx][lrow][lcol    ] = v.x;
            As[load_idx][lrow][lcol + 1] = v.y;
            As[load_idx][lrow][lcol + 2] = v.z;
            As[load_idx][lrow][lcol + 3] = v.w;
        }
        #pragma unroll
        for (int i = 0; i < B_LOADS; ++i) {
            const int lrow = bRow + i * bStride;
            const int lcol = bCol4 * 4;
            const int grow = ko + lrow;
            const int gcol = blockIdx.x * BN + lcol;
            float4 v = {};
            if (grow < K) {
                if (gcol + 3 < N) {
                    v = *reinterpret_cast<const float4*>(&B[grow * N + gcol]);
                } else {
                    v.x = (gcol     < N) ? B[grow * N + gcol    ] : 0.f;
                    v.y = (gcol + 1 < N) ? B[grow * N + gcol + 1] : 0.f;
                    v.z = (gcol + 2 < N) ? B[grow * N + gcol + 2] : 0.f;
                    v.w = (gcol + 3 < N) ? B[grow * N + gcol + 3] : 0.f;
                }
            }
            Bs[load_idx][lrow][lcol    ] = v.x;
            Bs[load_idx][lrow][lcol + 1] = v.y;
            Bs[load_idx][lrow][lcol + 2] = v.z;
            Bs[load_idx][lrow][lcol + 3] = v.w;
        }

        // Compute tile (ko - BK) using As[comp_idx], Bs[comp_idx]
        #pragma unroll
        for (int ki = 0; ki < BK; ++ki) {
            #pragma unroll
            for (int m = 0; m < TM; ++m)
                regA[m] = As[comp_idx][threadRow * TM + m][ki];
            #pragma unroll
            for (int n = 0; n < TN; ++n)
                regB[n] = Bs[comp_idx][ki][threadCol * TN + n];
            #pragma unroll
            for (int m = 0; m < TM; ++m)
                #pragma unroll
                for (int n = 0; n < TN; ++n)
                    regC[m][n] += regA[m] * regB[n];
        }

        __syncthreads();
        comp_idx = 1 - comp_idx;
    }

    // ── Final compute: last tile (no more prefetch) ──
    #pragma unroll
    for (int ki = 0; ki < BK; ++ki) {
        #pragma unroll
        for (int m = 0; m < TM; ++m)
            regA[m] = As[comp_idx][threadRow * TM + m][ki];
        #pragma unroll
        for (int n = 0; n < TN; ++n)
            regB[n] = Bs[comp_idx][ki][threadCol * TN + n];
        #pragma unroll
        for (int m = 0; m < TM; ++m)
            #pragma unroll
            for (int n = 0; n < TN; ++n)
                regC[m][n] += regA[m] * regB[n];
    }

    // ── Writeback TM×TN results using float4 ──
    #pragma unroll
    for (int m = 0; m < TM; ++m) {
        const int grow = cRowBase + m;
        if (grow >= M) continue;
        #pragma unroll
        for (int n = 0; n < TN; n += 4) {
            const int gcol = cColBase + n;
            if (gcol + 3 < N) {
                float4 old = *reinterpret_cast<const float4*>(&C[grow * N + gcol]);
                float4 out;
                out.x = alpha * regC[m][n    ] + beta * old.x;
                out.y = alpha * regC[m][n + 1] + beta * old.y;
                out.z = alpha * regC[m][n + 2] + beta * old.z;
                out.w = alpha * regC[m][n + 3] + beta * old.w;
                *reinterpret_cast<float4*>(&C[grow * N + gcol]) = out;
            } else {
                for (int ni = 0; ni < 4; ++ni)
                    if (gcol + ni < N)
                        C[grow * N + gcol + ni] =
                            alpha * regC[m][n + ni] + beta * C[grow * N + gcol + ni];
            }
        }
    }
}

// Launch wrapper
inline void launch_sgemm_v5_double_buffer(int M, int N, int K,
                                            float alpha, const float* A, const float* B,
                                            float beta,  float* C)
{
    constexpr int BM = 128, BN = 128, BK = 8;
    constexpr int TM = 8,   TN = 8;
    constexpr int BLOCK_X = BN / TN;   // 16
    constexpr int BLOCK_Y = BM / TM;   // 16
    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_v5_double_buffer<BM, BN, BK, TM, TN><<<grid, block>>>(
        M, N, K, alpha, A, B, beta, C);
}
