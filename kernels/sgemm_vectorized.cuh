#pragma once
#include <cuda_runtime.h>

// ─── Kernel 3: Register Tiling + Vectorized Loads ────────────────────────────
//
// Two key upgrades over kernel 2:
//
// 1. Register tiling (thread-level blocking):
//    Each thread computes TM×TN outputs instead of 1.
//    Load TM values of A and TN values of B into registers, then compute
//    the TM×TN outer product — TM*TN MADs from only TM+TN shared loads.
//    Arithmetic intensity per shared-load: TM*TN/(TM+TN) = 4× at TM=TN=8.
//
// 2. Vectorized global→shared loads (float4 = 128-bit LDG):
//    4 floats per load instruction → better memory-level parallelism,
//    lower instruction count, higher effective bandwidth utilization.
//
// Block tile  : BM×BN output elements per block  (128×128)
// K-step      : BK inner dimension per outer iter (8)
// Thread tile : TM×TN outputs per thread          (8×8)
// Block size  : (BM/TM)×(BN/TN) = 16×16 = 256 threads
//
// Shared memory: (BM*BK + BK*BN) * 4 = (1024+1024)*4 = 8 KB / block
// Registers/thread: TM+TN+TM*TN = 80 floats
//
// Expected: ~50–70% of cuBLAS on large square matrices.

template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_vectorized(int M, int N, int K,
                                  float alpha,
                                  const float* __restrict__ A,   // [M, K] row-major
                                  const float* __restrict__ B,   // [K, N] row-major
                                  float beta,
                                  float* __restrict__ C)         // [M, N] row-major
{
    static_assert(BK % 4 == 0, "BK must be divisible by 4");
    static_assert(BN % 4 == 0, "BN must be divisible by 4");
    static_assert(TN % 4 == 0, "TN must be divisible by 4 (for vectorized writeback)");

    constexpr int THREADS = (BM / TM) * (BN / TN);   // 256

    // ── Shared memory tiles ──
    __shared__ float As[BM][BK];   // 128×8
    __shared__ float Bs[BK][BN];   // 8×128

    // ── Flatten thread index ──
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // ── Which output sub-tile does this thread own? ──
    const int threadRow = tid / (BN / TN);   // 0..BM/TM-1  (row group)
    const int threadCol = tid % (BN / TN);   // 0..BN/TN-1  (col group)

    // ── Global output origin for this thread ──
    const int cRowBase = blockIdx.y * BM + threadRow * TM;
    const int cColBase = blockIdx.x * BN + threadCol * TN;

    // ── Accumulators & register tiles ──
    float regC[TM][TN] = {};
    float regA[TM];
    float regB[TN];

    // ── Load pattern: divide tile elements evenly across threads ──
    //
    // A tile (BM×BK): each thread loads 4 floats = 1 float4
    //   BM*BK / (THREADS*4) float4s per thread
    //   Arrange threads as: (BM×BK/4) elements split by tid
    constexpr int A_LOADS = (BM * BK) / (THREADS * 4);
    constexpr int B_LOADS = (BK * BN) / (THREADS * 4);

    // Thread position for loading As  (as float4 rows of width BK/4)
    const int aRow    = tid / (BK / 4);        // which row of As
    const int aCol4   = tid % (BK / 4);        // which float4 in that row
    const int aStride = THREADS / (BK / 4);    // row stride between loads

    // Thread position for loading Bs  (as float4 rows of width BN/4)
    const int bRow    = tid / (BN / 4);
    const int bCol4   = tid % (BN / 4);
    const int bStride = THREADS / (BN / 4);

    // ── Outer loop over K ──
    for (int ko = 0; ko < K; ko += BK) {

        // ── Load BM×BK tile of A → As (float4) ──
        // A[blockIdx.y*BM + aRow + i*aStride][ko + aCol4*4 .. +3]
        #pragma unroll
        for (int i = 0; i < A_LOADS; ++i) {
            const int lrow = aRow + i * aStride;       // local row in tile
            const int lcol = aCol4 * 4;                // local col in tile
            const int grow = blockIdx.y * BM + lrow;   // global row of A
            const int gcol = ko + lcol;                // global col of A
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
            As[lrow][lcol    ] = v.x;
            As[lrow][lcol + 1] = v.y;
            As[lrow][lcol + 2] = v.z;
            As[lrow][lcol + 3] = v.w;
        }

        // ── Load BK×BN tile of B → Bs (float4) ──
        // B[ko + bRow + i*bStride][blockIdx.x*BN + bCol4*4 .. +3]
        #pragma unroll
        for (int i = 0; i < B_LOADS; ++i) {
            const int lrow = bRow + i * bStride;
            const int lcol = bCol4 * 4;
            const int grow = ko + lrow;                    // global row of B
            const int gcol = blockIdx.x * BN + lcol;      // global col of B
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

        // ── Inner product over BK: each step = outer product TM×TN ──
        #pragma unroll
        for (int ki = 0; ki < BK; ++ki) {
            // Load TM values from As column ki
            #pragma unroll
            for (int m = 0; m < TM; ++m)
                regA[m] = As[threadRow * TM + m][ki];

            // Load TN values from Bs row ki
            #pragma unroll
            for (int n = 0; n < TN; ++n)
                regB[n] = Bs[ki][threadCol * TN + n];

            // Outer product accumulate
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
inline void launch_sgemm_vectorized(int M, int N, int K,
                                     float alpha, const float* A, const float* B,
                                     float beta,  float* C)
{
    constexpr int BM = 128, BN = 128, BK = 8;
    constexpr int TM = 8,   TN = 8;
    constexpr int BLOCK_X = BN / TN;   // 16
    constexpr int BLOCK_Y = BM / TM;   // 16
    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_vectorized<BM, BN, BK, TM, TN><<<grid, block>>>(
        M, N, K, alpha, A, B, beta, C);
}
