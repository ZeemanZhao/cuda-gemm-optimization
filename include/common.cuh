#pragma once
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// ─── Error checking ───────────────────────────────────────────────────────────

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

#define CUBLAS_CHECK(call)                                                  \
    do {                                                                    \
        cublasStatus_t st = (call);                                         \
        if (st != CUBLAS_STATUS_SUCCESS) {                                  \
            fprintf(stderr, "cuBLAS error at %s:%d — code %d\n",           \
                    __FILE__, __LINE__, (int)st);                           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// ─── GPU timer (cudaEvent based) ─────────────────────────────────────────────

struct GpuTimer {
    cudaEvent_t start_, stop_;
    GpuTimer()  { cudaEventCreate(&start_); cudaEventCreate(&stop_); }
    ~GpuTimer() { cudaEventDestroy(start_); cudaEventDestroy(stop_); }
    void start() { cudaEventRecord(start_); }
    // Returns elapsed milliseconds
    float stop() {
        cudaEventRecord(stop_);
        cudaEventSynchronize(stop_);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, start_, stop_);
        return ms;
    }
};

// ─── Matrix utilities ─────────────────────────────────────────────────────────

// Row-major: element (i,j) is at ptr[i*cols + j]
inline void mat_init_rand(float* ptr, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i)
        ptr[i] = (float)rand() / RAND_MAX - 0.5f;
}

inline void mat_init_zero(float* ptr, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i)
        ptr[i] = 0.f;
}

// Max absolute difference between two matrices
inline float max_abs_diff(const float* a, const float* b, int n) {
    float diff = 0.f;
    for (int i = 0; i < n; ++i)
        diff = fmaxf(diff, fabsf(a[i] - b[i]));
    return diff;
}

// GFLOPS for matmul: 2*M*N*K ops
inline double gflops(int M, int N, int K, double ms) {
    return 2.0 * M * N * K / (ms * 1e6);
}
