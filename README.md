# CUDA GEMM Optimization on RTX 4060

**English** | [中文](README.zh.md)

A step-by-step optimization study of single-precision matrix multiplication (SGEMM) on NVIDIA Ada Lovelace, written from scratch and benchmarked against cuBLAS.

This is a **learning project**: each kernel does just one optimization (shared-memory tiling, register tiling, vectorized loads, bank-conflict avoidance, double buffering, async copy), so the impact of every step can be attributed individually with Nsight Compute metrics.

v4 and v5 are **two optimizations that didn't pan out** — well-known wins on older architectures, but they regress on Ada (sm_89). They stay in the repo because the failure modes themselves are worth understanding. v6 then adds `cp.async` and becomes the fastest kernel — but the *reason* it wins (latency hiding, not the occupancy recovery you'd expect) is the most interesting result here.

![Performance bars at N=4096](docs/figures/performance_bars.png)

---

## Headline Numbers

| Kernel | GFLOPS @ M=N=K=4096 | Speedup vs Naive | % of cuBLAS |
|---|---:|---:|---:|
| v0 — Naive | 869 | 1.00× | 9.0% |
| v1 — Shared-memory tiling | 1,116 | 1.29× | 11.5% |
| v2 — Register tiling | 5,191 | 5.98× | 53.7% |
| v3 — + float4 vectorized loads | 7,683 | 8.85× | 79.5% |
| v4 — + 1-padding (negative result) | 7,268 | 8.37× | 75.2% |
| v5 — Double buffering (negative result) | 4,781 | 5.50× | 49.4% |
| **v6 — + cp.async pipeline** ⭐ | **8,708** | **10.02×** | **90.1%** |
| cuBLAS (FP32 SIMT, autotuned) | 9,668 | 11.13× | 100% |

Best handwritten kernel (**v6, cp.async**) reaches **90% of cuBLAS** — and this cuBLAS call is *itself* FP32 SIMT (its 9.7 TFLOPS is ~64% of the FP32 peak; a TF32 Tensor Core path would run far above the ~15 TFLOPS FP32 ceiling, which we don't see). So the remaining ~10% is **better FP32-SIMT engineering** — shape autotuning, warp-level tiling, swizzled layouts, L2-aware threadblock rasterization — *not* a different math unit. The surprise within v6 itself: its gain over v5 is **latency hiding, not occupancy** — explained in the v6 section below.

---

## Hardware

| Component | Spec |
|---|---|
| GPU | NVIDIA RTX 4060 Laptop (Ada Lovelace, sm_89) |
| FP32 peak | ~15.1 TFLOPS |
| Memory bandwidth | ~272 GB/s (GDDR6) |
| Shared memory / SM | 100 KB unified L1+shared (configurable) |
| Registers / SM | 65,536 × 32-bit |
| Toolchain | CUDA 12.4, Nsight Compute 2024.1.1, WSL2 (Ubuntu) |

> Note: the cuBLAS call here is `cublasSgemm` in the default math mode. Measured throughput (~9.7 TFLOPS, ~64% of FP32 peak) confirms it runs an **FP32-SIMT** kernel on this device — a TF32 Tensor Core path would land far above the ~15 TFLOPS FP32 ceiling. TF32 tensor cores *are* available (via `cublasMath` / the `NVIDIA_TF32_OVERRIDE` env var) and are a separate, higher ceiling at reduced precision — out of scope here.

---

## Kernel Versions

All kernels share the same launch convention: `C = α · A·B + β · C` with row-major storage and arbitrary M / N / K. Block tile, K step, and thread tile sizes are template parameters; the launch wrapper picks the configuration.

| Version | File | Optimization | Result |
|---|---|---|---|
| v0 | `kernels/sgemm_v0_naive.cuh` | One thread per output element; full K-loop directly from global memory. | baseline |
| v1 | `kernels/sgemm_v1_tiling.cuh` | Shared-memory tile (TILE=32); cooperative loading; classic blocked matmul. | +30% over v0 |
| v2 | `kernels/sgemm_v2_register_tile.cuh` | 8×8 output tile per thread; BM=BN=128, BK=8; **scalar** global → shared loads. 64 register accumulators enable ILP across independent FMAs. | **5.0× over v1** |
| v3 | `kernels/sgemm_v3_vectorized.cuh` | v2 + `float4` vectorized global → shared loads (128-bit LDG). Halves load instruction count. | **1.36× over v2**, float4 milestone |
| v4 | `kernels/sgemm_v4_bank_conflict_free.cuh` | v3 + `+1` shared-memory padding to break shared-bank conflict pattern. | **−3% on Ada** (negative) |
| v5 | `kernels/sgemm_v5_double_buffer.cuh` | v3 + double-buffered shared tiles to overlap global load with compute. | **−37% on Ada** (negative) |
| v6 | `kernels/sgemm_v6_cpasync.cuh` | v3 compute core + `cp.async` (`cuda::memcpy_async` + thread-scope `cuda::pipeline`) on the global→shared load path; 2-stage double buffer. | **+13% over v3**, best result (90% cuBLAS) |

### Why v4 regresses on Ada

The `+1` padding shifts shared-memory row stride from 16-byte aligned (8 / 128 floats) to non-aligned (9 / 129 floats). Original `float4` shared **stores** must degrade to four scalar stores, quadrupling instruction count on the LSU pipe. On Ada (sm_89), the saved bank-conflict wavefronts are smaller than the added store overhead, producing a net regression. On older architectures (Volta / Turing) where shared-bank arbitration is more expensive, the same change would yield a positive result. **Architecture-dependent optimization is the lesson.**

### Why v5 regresses on Ada

Software double buffering doubles per-block shared memory (8 KB → 16 KB), halving the concurrent blocks per SM and cutting occupancy. Without hardware-async copy (`cp.async`, sm_80+), nvcc can only do limited load/compute overlap between two software-managed buffers. The occupancy loss outweighs the latency hidden. The natural follow-up is `cp.async` + multi-stage pipeline — implemented as **v6 below**, with a result that overturns the occupancy hypothesis.

### Why v6 (cp.async) wins — and why the reason is surprising

v6 keeps v3's compute core unchanged and only swaps the global→shared load path to `cuda::memcpy_async` (lowering to `cp.async` / LDGSTS on sm_80+), driven by a thread-scope `cuda::pipeline` with a 2-stage buffer.

The hypothesis going in was that cp.async would **recover the occupancy** v5 lost, by removing the register-resident prefetch staging. Nsight Compute says that is **not** what happened:

| Metric @ 4096 | v3 | v5 | v6 |
|---|---:|---:|---:|
| registers / thread | 128 | 149 | 157 |
| achieved occupancy | 33.0% | 16.7% | **16.6%** |
| shared bank conflicts | 268 M | 268 M | 268 M |
| `long_scoreboard` stall | 1.66 | 1.88 | **0.075** |

v6's occupancy is **identical to v5's** (16.6%) and its register count is actually *higher* — the occupancy thesis failed outright. What won instead was **latency hiding**: cp.async overlaps the global→shared transfer with compute, collapsing the `long_scoreboard` (global-memory wait) stall from 1.88 to ~0.075.

Because v5 and v6 have **the same occupancy and the same bank conflicts**, the +82% jump from v5 → v6 is attributable almost entirely to that overlap. The lesson: `cp.async` provides **two independent benefits — register-bypass (→ occupancy) and async overlap (→ latency hiding)** — and on this kernel only the second one materialized. Low occupancy is not fatal when the latency it would have hidden is hidden another way. The bottleneck has now shifted from global-memory latency to the **shared-memory bank conflicts** (still 268 M) — the target for a future swizzled-layout v7.

---

### Performance scaling across matrix sizes

![GFLOPS vs matrix size](docs/figures/performance_lines.png)

---

## Build & Run

```bash
make                    # builds ./benchmark
./benchmark             # runs all sizes, prints comparison table
make clean              # removes build artifacts
```

Requirements: CUDA toolkit (≥ 11.8), `nvcc` and `cublas` on the linker path. The Makefile targets `sm_89`; edit `ARCH` for other GPUs.

### Profiling

`.ncu-rep` reports for every version are checked into `nsight_reports/`. View any of them with:

```bash
# CLI section view
ncu --import nsight_reports/v3_vectorized_4096.ncu-rep --section SpeedOfLight
ncu --import nsight_reports/v3_vectorized_4096.ncu-rep --section MemoryWorkloadAnalysis
ncu --import nsight_reports/v3_vectorized_4096.ncu-rep --section WarpStateStats
```

To regenerate a report:

```bash
ncu --set full --kernel-name regex:sgemm_v3_vectorized \
    --launch-skip 39 --launch-count 1 \
    -o nsight_reports/v3_vectorized_4096 -f \
    ./benchmark
```

`--launch-skip 39 --launch-count 1` skips the smaller-size warmups and profiles the first invocation at M=N=K=4096 (the steady-state size).

For GUI inspection, open `.ncu-rep` files in Nsight Compute on Windows / Linux desktop.

---

## Methodology

- **Reference**: cuBLAS `cublasSgemm` is the ground truth. Because cuBLAS expects column-major storage, the wrapper computes `Cᵀ = Bᵀ × Aᵀ` (a permuted call), which yields the correct row-major `C` without any explicit transpose.
- **Timing**: `cudaEvent_t` start/stop pairs (GPU-side, no host-launch overhead). Three untimed warmup iterations precede ten timed iterations; the average is reported.
- **Verification**: each kernel's output is compared to the cuBLAS result element-wise; runs warn if `max_abs_diff > 1e-2` (loose tolerance because `--use_fast_math` is enabled).
- **GFLOPS**: `2 · M · N · K / time_seconds / 1e9` — counts each FMA as two FLOPs.

---

## Repository Layout

```
.
├── benchmark.cu                              # benchmark driver
├── kernels/
│   ├── sgemm_v0_naive.cuh
│   ├── sgemm_v1_tiling.cuh
│   ├── sgemm_v2_register_tile.cuh
│   ├── sgemm_v3_vectorized.cuh
│   ├── sgemm_v4_bank_conflict_free.cuh
│   ├── sgemm_v5_double_buffer.cuh
│   └── sgemm_v6_cpasync.cuh
├── tests/
│   └── test_v6_correctness.cu                # v6 correctness vs cuBLAS
├── include/
│   └── common.cuh                            # CUDA_CHECK / CUBLAS_CHECK, GpuTimer, gflops()
├── nsight_reports/                           # .ncu-rep per version, archived
├── Makefile
├── LICENSE
└── README.md
```

---

## Out of Scope

Known but not done in this project:

- **Tensor Core / WMMA / `mma.sync`** — a separate, higher ceiling at reduced precision (TF32/FP16). *Not* the reason cuBLAS beats v6 here: this benchmark's cuBLAS is FP32 SIMT (see the Hardware note). A handwritten tensor-core kernel is its own project.
- **Swizzled shared layout (v7)** — XOR-permuted column addresses to remove the 268 M shared-bank conflicts that are now v6's dominant bottleneck. Planned next.
- **Warp specialization** (producer / consumer split inside one CTA).
- **Template-based autotuning** — searching the (BM, BN, BK, TM, TN) space per (M, N, K, sm_xx) at compile or runtime.

---

## References

- V. Volkov, *"Better Performance at Lower Occupancy"*, GTC 2010 — canonical argument that ILP can substitute for thread-level occupancy.
- NVIDIA CUDA C++ Programming Guide — memory model, bank conflicts, vectorized memory access.
- NVIDIA Nsight Compute Documentation — metric reference (`sm__throughput`, `smsp__warp_issue_stalled_*`, `l1tex__t_bytes_*`).
- CUTLASS (`github.com/NVIDIA/cutlass`) — production-grade reference for hierarchical tiling, swizzling, async pipelines.
- Lei Mao's CUDA blog (`leimao.github.io`) — accessible step-by-step GEMM optimization writeups.

---

## License

MIT — see [LICENSE](LICENSE).
