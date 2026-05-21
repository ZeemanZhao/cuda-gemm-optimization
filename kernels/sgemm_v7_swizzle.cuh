#pragma once
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda/pipeline>

// ─── Kernel v7: cp.async + Bs XOR-swizzle (async global→shared) double buffering ───────────────
//
// Identical to v6 (cp.async + thread-scope cuda::pipeline, NSTAGES=2 double buffer),
// PLUS an XOR swizzle on the Bs shared layout (swz_col below) to attack the 268M shared
// bank conflict v6 inherited from v3. swz_col maps each Bs logical column to a physical
// column, applied at BOTH the cp.async store and the compute read (same function).
//
// The Bs read is 4-way bank-conflicting: a warp's 16 float4 columns land in only 4 of
// the 8 float4 bank-groups; swz_col spreads them across all 8 -> 2-way. 2-way (not 0) is
// the floor for this 16x2 warp shape; zero would need a warp reshape (future work).
// As is left unswizzled (2-way + mostly broadcast, not worth it).
//
// ASSUMES tile-aligned dims: M % BM == 0, N % BN == 0, K % BK == 0. No ragged-edge handling.

// float4-granularity XOR swizzle for Bs columns: perm(c4) = c4 ^ (c4 >> 3)
// (c4 = float4 index within the 128-wide row, 0..31). Bijective, row-independent,
// preserves 16-byte alignment (permutes whole float4s). Spreads the warp's 16 float4
// reads from 4 bank-groups to 8 -> 4-way down to 2-way.
__device__ __forceinline__ int swz_col(int col) {
    const int c4 = col >> 2;          // col / 4
    const int w  = col & 3;           // col % 4
    const int p  = c4 ^ (c4 >> 3);    // XOR swizzle on the float4 index
    return (p << 2) | w;              // physical col = perm(c4)*4 + w
}

template <int BM, int BN, int BK, int TM, int TN, int NSTAGES>
__global__ void sgemm_v7_swizzle(int M, int N, int K,
                                  float alpha,
                                  const float* __restrict__ A,   // [M,K] row-major
                                  const float* __restrict__ B,   // [K,N] row-major
                                  float beta,
                                  float* __restrict__ C)         // [M,N] row-major
{
    static_assert(BK % 4 == 0, "BK must be divisible by 4");
    static_assert(BN % 4 == 0, "BN must be divisible by 4");
    static_assert(TN % 4 == 0, "TN must be divisible by 4 (vectorized writeback)");
    static_assert(NSTAGES >= 2, "need >=2 stages for overlap");

    constexpr int THREADS = (BM / TM) * (BN / TN);   // 256

    __shared__ float As[NSTAGES][BM][BK];
    __shared__ float Bs[NSTAGES][BK][BN];

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

    const int numTiles = K / BK;

    auto pipe = cuda::make_pipeline();   // thread-scope pipeline (no shared mbarrier)

    // ── Prologue: kick off cp.async for the first NSTAGES tiles (fill the ring) ──
    #pragma unroll
    for (int s = 0; s < NSTAGES; ++s) {
        pipe.producer_acquire();
        if (s < numTiles) {
            const int ko = s * BK;
            #pragma unroll
            for (int i = 0; i < A_LOADS; ++i) {
                const int lrow = aRow + i * aStride;
                const int lcol = aCol4 * 4;
                const int grow = blockIdx.y * BM + lrow;
                cuda::memcpy_async(&As[s][lrow][lcol],
                                   &A[grow * K + (ko + lcol)],
                                   cuda::aligned_size_t<16>(16), pipe);
            }
            #pragma unroll
            for (int i = 0; i < B_LOADS; ++i) {
                const int lrow = bRow + i * bStride;
                const int lcol = bCol4 * 4;
                const int grow = ko + lrow;
                cuda::memcpy_async(&Bs[s][lrow][swz_col(lcol)],
                                   &B[grow * N + (blockIdx.x * BN + lcol)],
                                   cuda::aligned_size_t<16>(16), pipe);
            }
        }
        pipe.producer_commit();
    }

    // ── Steady state: compute tile `compute`, then refill its buffer with `fetch` ──
    for (int compute = 0, fetch = NSTAGES; compute < numTiles; ++compute, ++fetch) {
        // Wait until only NSTAGES-1 groups remain in flight → oldest (this tile) is ready.
        cuda::pipeline_consumer_wait_prior<NSTAGES - 1>(pipe);
        __syncthreads();   // cross-thread: whole tile (all threads' slices) visible

        const int cstage = compute % NSTAGES;
        #pragma unroll
        for (int ki = 0; ki < BK; ++ki) {
            #pragma unroll
            for (int m = 0; m < TM; ++m)
                regA[m] = As[cstage][threadRow * TM + m][ki];
            #pragma unroll
            for (int f4 = 0; f4 < TN / 4; ++f4) {
                const float4 vb = *reinterpret_cast<const float4*>(
                    &Bs[cstage][ki][swz_col(threadCol * TN + f4 * 4)]);
                regB[f4*4 + 0] = vb.x;
                regB[f4*4 + 1] = vb.y;
                regB[f4*4 + 2] = vb.z;
                regB[f4*4 + 3] = vb.w;
            }
            #pragma unroll
            for (int m = 0; m < TM; ++m)
                #pragma unroll
                for (int n = 0; n < TN; ++n)
                    regC[m][n] += regA[m] * regB[n];
        }

        pipe.consumer_release();   // free this buffer for reuse
        __syncthreads();           // ensure all threads done reading before overwrite

        // Refill the just-freed buffer with the future tile `fetch`.
        pipe.producer_acquire();
        if (fetch < numTiles) {
            const int fstage = compute % NSTAGES;
            const int ko = fetch * BK;
            #pragma unroll
            for (int i = 0; i < A_LOADS; ++i) {
                const int lrow = aRow + i * aStride;
                const int lcol = aCol4 * 4;
                const int grow = blockIdx.y * BM + lrow;
                cuda::memcpy_async(&As[fstage][lrow][lcol],
                                   &A[grow * K + (ko + lcol)],
                                   cuda::aligned_size_t<16>(16), pipe);
            }
            #pragma unroll
            for (int i = 0; i < B_LOADS; ++i) {
                const int lrow = bRow + i * bStride;
                const int lcol = bCol4 * 4;
                const int grow = ko + lrow;
                cuda::memcpy_async(&Bs[fstage][lrow][swz_col(lcol)],
                                   &B[grow * N + (blockIdx.x * BN + lcol)],
                                   cuda::aligned_size_t<16>(16), pipe);
            }
        }
        pipe.producer_commit();
    }

    // ── Writeback TM×TN results (float4), tile-aligned so no boundary guards ──
    #pragma unroll
    for (int m = 0; m < TM; ++m) {
        const int grow = cRowBase + m;
        #pragma unroll
        for (int n = 0; n < TN; n += 4) {
            const int gcol = cColBase + n;
            float4 old = *reinterpret_cast<const float4*>(&C[grow * N + gcol]);
            float4 out;
            out.x = alpha * regC[m][n    ] + beta * old.x;
            out.y = alpha * regC[m][n + 1] + beta * old.y;
            out.z = alpha * regC[m][n + 2] + beta * old.z;
            out.w = alpha * regC[m][n + 3] + beta * old.w;
            *reinterpret_cast<float4*>(&C[grow * N + gcol]) = out;
        }
    }
}

// Launch wrapper
inline void launch_sgemm_v7_swizzle(int M, int N, int K,
                                     float alpha, const float* A, const float* B,
                                     float beta,  float* C)
{
    constexpr int BM = 128, BN = 128, BK = 8;
    constexpr int TM = 8,   TN = 8;
    constexpr int NSTAGES = 2;   // ← change to 3 for the deeper-pipeline experiment (Task 5)
    constexpr int BLOCK_X = BN / TN;   // 16
    constexpr int BLOCK_Y = BM / TM;   // 16
    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_v7_swizzle<BM, BN, BK, TM, TN, NSTAGES><<<grid, block>>>(
        M, N, K, alpha, A, B, beta, C);
}
