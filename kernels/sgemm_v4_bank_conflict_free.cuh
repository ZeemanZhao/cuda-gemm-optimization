#pragma once
#include <cuda_runtime.h>

// ─── Kernel v4: Bank-Conflict-Free Shared Memory (via +1 padding) ─────────────
//
// Same algorithm as v3 (register tile + float4 vectorized loads), but the
// shared-memory tiles are padded by 1 element along their inner dimension to
// break shared-memory bank-conflict patterns.
//
// Why bank conflicts happen in v3:
//   - Shared memory has 32 banks, 4-byte stride (one float per bank-cycle).
//   - In v3, the inner loop reads:
//        regA[m] = As[threadRow * TM + m][ki];
//        regB[n] = Bs[ki][threadCol * TN + n];
//   - For Bs[ki][threadCol*8 + n], threads with threadCol differing by 4
//     hit the same bank (8 floats/stride × 4 = 32 banks repeat) → 8-way conflict.
//   - For As[row][ki], threads with threadRow differing by 4 alias too.
//
// The fix — pad each row by 1 float:
//        __shared__ float As[BM][BK + 1];   // [128][9] instead of [128][8]
//        __shared__ float Bs[BK][BN + 1];   // [8][129] instead of [8][128]
//   The +1 shifts row strides off the 32-bank repeat, so threads that used
//   to alias now land on different banks. This breaks the broadcast/8-way
//   pattern at the cost of (BM + BK) × 4 = 544 extra bytes / block.
//
// Expected ncu signal: `Shared Memory Bank Conflicts` drops to (near) 0.
// Expected speedup: 5–15% over v3 on Ada (sm_89).

template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_v4_bank_conflict_free(int M, int N, int K,
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

    // ── Padded shared memory tiles ──
    __shared__ float As[BM][BK + 1];   // 128 × 9 (was 128 × 8 in v3)
    __shared__ float Bs[BK][BN + 1];   // 8 × 129 (was 8 × 128 in v3)

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;

    const int threadRow = tid / (BN / TN);   // 0..15
    const int threadCol = tid % (BN / TN);   // 0..15

    const int cRowBase = blockIdx.y * BM + threadRow * TM;
    const int cColBase = blockIdx.x * BN + threadCol * TN;

    float regC[TM][TN] = {};
    float regA[TM];
    float regB[TN];

    // Load pattern: divide tile elements evenly across threads, one float4 each
    constexpr int A_LOADS = (BM * BK) / (THREADS * 4);
    constexpr int B_LOADS = (BK * BN) / (THREADS * 4);

    const int aRow    = tid / (BK / 4);
    const int aCol4   = tid % (BK / 4);
    const int aStride = THREADS / (BK / 4);

    const int bRow    = tid / (BN / 4);
    const int bCol4   = tid % (BN / 4);
    const int bStride = THREADS / (BN / 4);

    for (int ko = 0; ko < K; ko += BK) {

        // ── Load BM×BK tile of A → As (float4) ──
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
            // NOTE: scalar stores into padded shared (cannot use float4 because
            //       row stride is BK+1=9, no longer 16-byte-aligned).
            As[lrow][lcol    ] = v.x;
            As[lrow][lcol + 1] = v.y;
            As[lrow][lcol + 2] = v.z;
            As[lrow][lcol + 3] = v.w;
        }

        // ── Load BK×BN tile of B → Bs (float4) ──
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
            Bs[lrow][lcol    ] = v.x;
            Bs[lrow][lcol + 1] = v.y;
            Bs[lrow][lcol + 2] = v.z;
            Bs[lrow][lcol + 3] = v.w;
        }

        __syncthreads();

        // ── Inner BK loop: each ki step is a TM×TN outer product ──
        #pragma unroll
        for (int ki = 0; ki < BK; ++ki) {
            #pragma unroll
            for (int m = 0; m < TM; ++m)
                regA[m] = As[threadRow * TM + m][ki];

            #pragma unroll
            for (int n = 0; n < TN; ++n)
                regB[n] = Bs[ki][threadCol * TN + n];

            #pragma unroll
            for (int m = 0; m < TM; ++m)
                #pragma unroll
                for (int n = 0; n < TN; ++n)
                    regC[m][n] += regA[m] * regB[n];
        }

        __syncthreads();
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
inline void launch_sgemm_v4_bank_conflict_free(int M, int N, int K,
                                                 float alpha, const float* A, const float* B,
                                                 float beta,  float* C)
{
    constexpr int BM = 128, BN = 128, BK = 8;
    constexpr int TM = 8,   TN = 8;
    constexpr int BLOCK_X = BN / TN;   // 16
    constexpr int BLOCK_Y = BM / TM;   // 16
    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_v4_bank_conflict_free<BM, BN, BK, TM, TN><<<grid, block>>>(
        M, N, K, alpha, A, B, beta, C);
}
