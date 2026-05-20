#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "include/common.cuh"
#include "kernels/sgemm_v6_cpasync.cuh"

static cublasHandle_t g_cublas;

// cuBLAS reference: C(M×N row-major) = A·B  via  C^T = B^T · A^T  (same trick as benchmark.cu)
static void cublas_ref(int M, int N, int K, float alpha,
                       const float* dA, const float* dB, float beta, float* dC) {
    CUBLAS_CHECK(cublasSgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K, &alpha, dB, N, dA, K, &beta, dC, N));
}

static float run_case(int M, int N, int K) {
    size_t szA = (size_t)M*K, szB = (size_t)K*N, szC = (size_t)M*N;
    float *hA = new float[szA], *hB = new float[szB];
    float *hRef = new float[szC], *hOut = new float[szC];
    srand(42);
    mat_init_rand(hA, M, K);
    mat_init_rand(hB, K, N);

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, szA*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, szB*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, szC*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, hA, szA*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, szB*sizeof(float), cudaMemcpyHostToDevice));

    const float alpha = 1.f, beta = 0.f;

    // reference
    CUDA_CHECK(cudaMemset(dC, 0, szC*sizeof(float)));
    cublas_ref(M, N, K, alpha, dA, dB, beta, dC);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hRef, dC, szC*sizeof(float), cudaMemcpyDeviceToHost));

    // v6 under test
    CUDA_CHECK(cudaMemset(dC, 0, szC*sizeof(float)));
    launch_sgemm_v6_cpasync(M, N, K, alpha, dA, dB, beta, dC);
    CUDA_CHECK(cudaGetLastError());          // catch launch-config errors
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hOut, dC, szC*sizeof(float), cudaMemcpyDeviceToHost));

    float err = max_abs_diff(hOut, hRef, (int)szC);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    delete[] hA; delete[] hB; delete[] hRef; delete[] hOut;
    return err;
}

int main() {
    CUBLAS_CHECK(cublasCreate(&g_cublas));
    const int sizes[] = { 256, 512 };   // tile-aligned (multiples of BM=BN=128, BK=8)
    int fails = 0;
    // 1e-2 = project's established bar (benchmark.cu). v6 compute is bit-identical to v3,
    // so any diff vs cuBLAS is pure SIMT-vs-cuBLAS accumulation + --use_fast_math, not a v6 bug.
    // We print the actual err (expect ~1e-4) to confirm.
    for (int s : sizes) {
        float err = run_case(s, s, s);
        bool ok = err < 1e-2f;
        printf("v6 %4dx%4dx%4d  max_abs_diff=%.2e  %s\n", s, s, s, err, ok ? "PASS" : "FAIL");
        if (!ok) ++fails;
    }
    cublasDestroy(g_cublas);
    if (fails) { printf("FAILED %d case(s)\n", fails); return 1; }
    printf("ALL PASS\n");
    return 0;
}
