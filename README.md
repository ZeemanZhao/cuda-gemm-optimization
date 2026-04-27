# CUDA GEMM Optimization

A step-by-step optimization study of single-precision matrix multiplication (SGEMM) on NVIDIA Ada Lovelace, written from scratch and benchmarked against cuBLAS.

The goal is pedagogical: each kernel isolates one optimization technique (shared-memory tiling, register tiling, vectorized loads, bank conflict avoidance, double buffering) so the performance impact of every step can be attributed and reasoned about with Nsight Compute metrics.

---

## Hardware

| Component | Spec |
|---|---|
| GPU | NVIDIA RTX 4060 Laptop (Ada Lovelace, sm_89) |
| FP32 peak | ~15.1 TFLOPS |
| Memory bandwidth | ~272 GB/s |
| Shared memory / SM | 64 KB (configurable) |
| Registers / SM | 65,536 × 32-bit |
| Toolchain | CUDA 12.4, Nsight Compute 2024.1.1, WSL2 (Ubuntu) |

> Note: cuBLAS in this benchmark is FP32 SGEMM. On Ampere+ hardware cuBLAS may internally dispatch to TF32 Tensor Cores depending on `cublasMath` settings, which explains why it remains a non-trivial ceiling for purely FP32-SIMT handwritten kernels.

---

## Current Status

Three kernels implemented; two more planned. All kernels share the same launch convention:
`C = α · A·B + β · C`, with row-major storage and arbitrary M / N / K.

| Version | File | What's new |
|---|---|---|
| v0 — Naive | `kernels/sgemm_naive.cuh` | One thread per output element; full K-loop directly from global memory. |
| v1 — Block tiling | `kernels/sgemm_tiling.cuh` | Shared-memory tile (TILE=32); cooperative loading; classic blocked matmul. |
| v3 — Register tiling + vectorized | `kernels/sgemm_vectorized.cuh` | 8×8 output tile per thread; BM=BN=128, BK=8; `float4` vectorized global→shared loads. |
| v2 — Register tiling (scalar) | _planned_ | Split from v3: register tiling alone, no `float4`, to isolate vectorization contribution. |
| v4 — Bank-conflict free | _planned_ | `+1` padding on shared tiles to break the 4-way bank conflict in the inner-product step. |
| v5 — Double buffering | _planned (stretch)_ | Two shared-memory tiles to overlap global load with compute on the next K-step. |

---

## Measured Performance

**Hardware**: RTX 4060 Laptop, CUDA 12.4. **Settings**: `α=1`, `β=0`, 3 warmup + 10 timed iterations, `cudaEvent` GPU-side timing. **Verification**: `max_abs_diff(out, cuBLAS) < 1e-2` per kernel per size.

| M=N=K | Naive (GFLOPS) | Tiling (GFLOPS) | VecReg (GFLOPS) | cuBLAS (GFLOPS) | VecReg / cuBLAS |
|---:|---:|---:|---:|---:|---:|
| 512  | 670  | 860  | 2,669 | 4,919 | 54.3% |
| 1024 | 709  | 892  | 4,854 | 6,299 | 77.1% |
| 2048 | 864  | 1,236 | 7,854 | 9,614 | 81.7% |
| 4096 | 858  | 1,106 | 7,587 | 9,343 | 81.2% |

**Headline numbers (M=N=K=4096)**:

- Final kernel reaches **7.59 TFLOPS**, **81.2% of cuBLAS**.
- Final-vs-naive speedup: **8.85×** at 4096 (max **9.1×** at 2048).
- The best handwritten FP32 SIMT kernel cannot exceed cuBLAS itself (which here measures 9.34 TFLOPS at 4096), since cuBLAS sets the practical FP32 ceiling on this device.

Final-results table will be updated once v4 and v5 land.

---

## Build & Run

```bash
make                    # builds ./benchmark
./benchmark             # runs all sizes, prints comparison table

make profile            # Nsight Compute pass with selected metrics
make clean              # removes build artifacts
```

Requirements: CUDA toolkit (≥ 11.8), Nsight Compute (≥ 2022.4 for Ada support), `nvcc` and `cublas` available on the linker path. The Makefile targets `sm_89`; edit `ARCH` for other GPUs.

For full Nsight reports (`.ncu-rep`):

```bash
ncu --set full -o nsight_reports/<version> ./benchmark
```

---

## Methodology

- **Reference**: cuBLAS `cublasSgemm` is the ground truth. Because cuBLAS expects column-major storage, the wrapper computes `Cᵀ = Bᵀ × Aᵀ` (a permuted call), which yields the correct row-major `C` without any explicit transpose.
- **Timing**: `cudaEvent_t` start/stop pairs (GPU-side, no host-launch overhead). Three untimed warmup iterations precede ten timed iterations; the average is reported.
- **Verification**: each kernel's output is compared to the cuBLAS result element-wise; runs flag a warning if `max_abs_diff > 1e-2` (loose tolerance because `--use_fast_math` is enabled).
- **GFLOPS**: `2 · M · N · K / time_seconds / 1e9` — counts each FMA as two FLOPs.

---

## Optimization Journey (so far)

### v0 → v1 (block tiling)

Naive arithmetic intensity is ~0.25 FLOPs/byte (each FMA reads two 4-byte operands from global memory), well below the device's roofline crossover (~58 FLOPs/byte). Block tiling raises this by a factor of `TILE` because every operand loaded into shared memory is reused `TILE` times before eviction. With `TILE=32`, theoretical traffic reduction is ~32×. Measured 4096-size speedup is only ~1.3×: the L2 cache already buffered a meaningful fraction of the naive kernel's repeated loads, narrowing the gap. This is a useful reminder that "the obvious bandwidth model" overestimates the gain when L2 is a free win at small enough working-set sizes.

### v1 → v3 (register tiling + vectorized loads)

Two changes happen together (split into v2 and v3 is on the roadmap):

1. **Register tiling**: each thread now computes an `8×8` output sub-tile, holding 64 accumulators in registers. `8+8 = 16` shared-memory loads feed `8×8 = 64` FMAs — arithmetic intensity per shared load is `4×` higher than `TILE=32` block tiling. Occupancy drops (more registers per thread), but instruction-level parallelism within each thread covers the latency, and the shared-memory pressure drops sharply.
2. **`float4` vectorization**: global → shared loads are issued as 128-bit transactions, halving the instruction count for the load phase and improving achieved bandwidth.

The combined kernel reaches ~7.6 TFLOPS at 4096, a roughly 7× jump from block tiling.

### Open questions (motivating v2/v4/v5)

- How much of the 7× jump is register tiling vs `float4`? — answered by v2 (split).
- The inner-product loop reads `Bs[ki][threadCol*TN + n]` with stride 8 floats across consecutive threads, which maps to a 4-way bank conflict on the 32-bank shared memory. Does eliminating that with `+1` padding (v4) yield the predicted 5–10% gain?
- With BK=8, the `__syncthreads()` between K-steps stalls every block. v5 (double buffering) overlaps the next tile's load with the current tile's compute. How close to cuBLAS does this get on this device?

---

## Repository Layout

```
.
├── benchmark.cu              # benchmark driver: cuBLAS reference, cudaEvent timing, verification
├── kernels/
│   ├── sgemm_naive.cuh       # v0
│   ├── sgemm_tiling.cuh      # v1
│   └── sgemm_vectorized.cuh  # v3 (register tile + float4)
├── include/
│   └── common.cuh            # CUDA_CHECK / CUBLAS_CHECK macros, GpuTimer, gflops()
├── nsight_reports/           # profiler outputs (text snapshots; .ncu-rep when generated)
├── Makefile
└── README.md
```

---

## Roadmap

- [x] Naive baseline (v0)
- [x] Shared-memory block tiling (v1)
- [x] Register tiling + `float4` vectorized loads (v3, combined)
- [ ] Split v2 (register tiling, scalar loads) from v3 to attribute speedup correctly
- [ ] Bank-conflict-free shared layout (v4)
- [ ] Double buffering (v5)
- [ ] Per-version Nsight Compute reports archived in `nsight_reports/`
- [ ] Final benchmark table and per-version metric annotations in this README

Out of scope for this study (acknowledged but not pursued): Tensor Core / WMMA / TF32 / FP16 paths, `cp.async` (asynchronous global → shared copies), warp specialization (producer / consumer split), and template-based autotuning. These are the main reasons cuBLAS is faster, and each is a substantial project on its own.

---

## References

- V. Volkov, *"Better Performance at Lower Occupancy"*, GTC 2010 — the canonical argument that ILP can substitute for thread-level occupancy.
- NVIDIA CUDA C++ Programming Guide — memory model, bank conflicts, vectorized memory access.
- NVIDIA Nsight Compute Documentation — metric definitions used in this study (`sm__throughput`, `l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second`, `smsp__warps_active`, etc.).
- CUTLASS (`github.com/NVIDIA/cutlass`) — production-grade reference for hierarchical tiling, swizzling, async pipelines.
- Lei Mao's CUDA blog (`leimao.github.io`) — accessible step-by-step GEMM optimization writeups.

---

## License

MIT — see [LICENSE](LICENSE).
