#pragma once
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda/pipeline>

// ─── Kernel v6: cp.async (async global→shared) double buffering ───────────────
//
// Same compute core as v3 (register tile + float4), but the global→shared load
// path uses cuda::memcpy_async (lowers to cp.async / LDGSTS on sm_80+) driven by
// a THREAD-SCOPE cuda::pipeline. The async copy bypasses the register file, so —
// unlike v5's manual double buffering — it does NOT spend extra registers holding
// the next tile across the inner loop. Goal: recover v5's lost occupancy
// (149 reg → 16.7% occ) back toward v3's 128 reg / 33% occ, keeping the overlap.
//
// Pipeline: NSTAGES shared buffers (ring). Keep NSTAGES-1 copies in flight via
//   cuda::pipeline_consumer_wait_prior<NSTAGES-1>. NSTAGES=2 == double buffer.
//   Change NSTAGES (launch wrapper) to experiment with deeper pipelines.
//
// ASSUMES tile-aligned dims: M % BM == 0, N % BN == 0, K % BK == 0.
//   All benchmark sizes satisfy this. No ragged-edge handling.

template <int BM, int BN, int BK, int TM, int TN, int NSTAGES>
__global__ void sgemm_v6_cpasync(int M, int N, int K,
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
                cuda::memcpy_async(&Bs[s][lrow][lcol],
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
            for (int n = 0; n < TN; ++n)
                regB[n] = Bs[cstage][ki][threadCol * TN + n];
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
                cuda::memcpy_async(&Bs[fstage][lrow][lcol],
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
inline void launch_sgemm_v6_cpasync(int M, int N, int K,
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
    sgemm_v6_cpasync<BM, BN, BK, TM, TN, NSTAGES><<<grid, block>>>(
        M, N, K, alpha, A, B, beta, C);
}
