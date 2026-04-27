#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "include/common.cuh"
#include "kernels/sgemm_naive.cuh"
#include "kernels/sgemm_tiling.cuh"
#include "kernels/sgemm_vectorized.cuh"

// ─── Config ───────────────────────────────────────────────────────────────────

static const int WARMUP  = 3;    // warmup iterations (not timed)
static const int REPEATS = 10;   // timed iterations

// Matrix sizes to benchmark
static const int SIZES[] = { 512, 1024, 2048, 4096 };
static const int N_SIZES = sizeof(SIZES) / sizeof(SIZES[0]);

// ─── Helpers ──────────────────────────────────────────────────────────────────

// Copy host matrix to device and zero-init output
static void prepare_device(const float* hA, const float* hB, float* hC,
                             float*& dA, float*& dB, float*& dC,
                             int M, int N, int K)
{
    size_t bytesA = (size_t)M * K * sizeof(float);
    size_t bytesB = (size_t)K * N * sizeof(float);
    size_t bytesC = (size_t)M * N * sizeof(float);
    CUDA_CHECK(cudaMalloc(&dA, bytesA));
    CUDA_CHECK(cudaMalloc(&dB, bytesB));
    CUDA_CHECK(cudaMalloc(&dC, bytesC));
    CUDA_CHECK(cudaMemcpy(dA, hA, bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, bytesB, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC, hC, bytesC, cudaMemcpyHostToDevice));
}

// Reset dC to host hC before each kernel run
static void reset_C(const float* hC, float* dC, int M, int N)
{
    CUDA_CHECK(cudaMemcpy(dC, hC, (size_t)M * N * sizeof(float),
                          cudaMemcpyHostToDevice));
}

// Time a kernel over WARMUP+REPEATS iterations; return avg ms
template <typename KernelFn>
static double time_kernel(KernelFn fn,
                           int M, int N, int K,
                           float alpha, float* dA, float* dB,
                           float beta,  float* dC, const float* hC_zero)
{
    GpuTimer timer;

    // Warmup
    for (int i = 0; i < WARMUP; ++i) {
        reset_C(hC_zero, dC, M, N);
        fn(M, N, K, alpha, dA, dB, beta, dC);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed runs
    reset_C(hC_zero, dC, M, N);
    timer.start();
    for (int i = 0; i < REPEATS; ++i) {
        fn(M, N, K, alpha, dA, dB, beta, dC);
    }
    float ms = timer.stop();
    return (double)ms / REPEATS;
}

// ─── cuBLAS SGEMM wrapper ─────────────────────────────────────────────────────
//
// cuBLAS assumes column-major storage. For row-major A (M×K) and B (K×N),
// we compute C^T = B^T × A^T, which gives the correct row-major result in C.
static cublasHandle_t g_cublas;

static void cublas_sgemm(int M, int N, int K,
                          float alpha, const float* dA, const float* dB,
                          float beta,  float* dC)
{
    // C (M×N, row-major) ≡ C^T (N×M, col-major)
    // C^T = B^T × A^T
    CUBLAS_CHECK(cublasSgemm(g_cublas,
                              CUBLAS_OP_N, CUBLAS_OP_N,
                              N, M, K,            // m, n, k of B^T × A^T
                              &alpha,
                              dB, N,              // B^T (N×K col-major) ldb=N
                              dA, K,              // A^T (K×M col-major) lda=K
                              &beta,
                              dC, N));            // C^T (N×M col-major) ldc=N
}

// ─── Main ─────────────────────────────────────────────────────────────────────

int main()
{
    CUBLAS_CHECK(cublasCreate(&g_cublas));

    printf("GPU: RTX 4060 Laptop  |  CUDA 12.4  |  sm_89\n");
    printf("alpha=1.0, beta=0.0,  repeats=%d\n\n", REPEATS);

    // Header
    printf("%-8s  %-12s  %-12s  %-12s  %-12s  %-12s  %-12s  %-8s\n",
           "Size",
           "Naive(ms)", "Naive(Gf)",
           "Tiling(ms)", "Tiling(Gf)",
           "VecReg(ms)", "VecReg(Gf)",
           "cuBLAS(Gf)");
    printf("%s\n", std::string(110, '-').c_str());

    const float alpha = 1.f, beta = 0.f;

    for (int si = 0; si < N_SIZES; ++si) {
        int M = SIZES[si], N = SIZES[si], K = SIZES[si];

        // ── Allocate host matrices ──
        size_t szA = (size_t)M * K, szB = (size_t)K * N, szC = (size_t)M * N;
        float* hA   = new float[szA];
        float* hB   = new float[szB];
        float* hC0  = new float[szC];   // zero, used as initial C
        float* hRef = new float[szC];   // cuBLAS reference result
        float* hOut = new float[szC];   // kernel result

        srand(42);
        mat_init_rand(hA, M, K);
        mat_init_rand(hB, K, N);
        mat_init_zero(hC0, M, N);

        // ── Device buffers ──
        float *dA, *dB, *dC;
        prepare_device(hA, hB, hC0, dA, dB, dC, M, N, K);

        // ── cuBLAS reference (generate gold standard) ──
        reset_C(hC0, dC, M, N);
        cublas_sgemm(M, N, K, alpha, dA, dB, beta, dC);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(hRef, dC, szC * sizeof(float),
                               cudaMemcpyDeviceToHost));

        // Helper: verify a kernel's output
        auto verify = [&](const char* name) {
            CUDA_CHECK(cudaMemcpy(hOut, dC, szC * sizeof(float),
                                   cudaMemcpyDeviceToHost));
            float err = max_abs_diff(hOut, hRef, (int)szC);
            if (err > 1e-2f)
                fprintf(stderr, "  [WARN] %s max_abs_diff=%.4f\n", name, err);
        };

        // ── Benchmark ──
        double ms_naive = 0, ms_tile = 0, ms_vec = 0, ms_cublas = 0;

        // Naive
        ms_naive = time_kernel(launch_sgemm_naive,
                               M, N, K, alpha, dA, dB, beta, dC, hC0);
        verify("naive");

        // Tiling
        ms_tile = time_kernel(launch_sgemm_tiling,
                               M, N, K, alpha, dA, dB, beta, dC, hC0);
        verify("tiling");

        // Vectorized
        ms_vec = time_kernel(launch_sgemm_vectorized,
                              M, N, K, alpha, dA, dB, beta, dC, hC0);
        verify("vectorized");

        // cuBLAS
        ms_cublas = time_kernel(cublas_sgemm,
                                 M, N, K, alpha, dA, dB, beta, dC, hC0);

        // ── Print row ──
        double gf_naive  = ms_naive  > 0 ? gflops(M,N,K,ms_naive)  : 0;
        double gf_tile   = gflops(M, N, K, ms_tile);
        double gf_vec    = gflops(M, N, K, ms_vec);
        double gf_cublas = gflops(M, N, K, ms_cublas);

        if (ms_naive > 0)
            printf("%-8d  %-12.3f  %-12.1f  %-12.3f  %-12.1f  %-12.3f  %-12.1f  %-8.1f\n",
                   M, ms_naive, gf_naive, ms_tile, gf_tile,
                   ms_vec, gf_vec, gf_cublas);
        else
            printf("%-8d  %-12s  %-12s  %-12.3f  %-12.1f  %-12.3f  %-12.1f  %-8.1f\n",
                   M, "(skip)", "(skip)", ms_tile, gf_tile,
                   ms_vec, gf_vec, gf_cublas);

        // ── Cleanup ──
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
        delete[] hA; delete[] hB; delete[] hC0;
        delete[] hRef; delete[] hOut;
    }

    cublasDestroy(g_cublas);
    return 0;
}
