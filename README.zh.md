# RTX 4060 上的 CUDA GEMM 优化

[English](README.md) | **中文**

在 NVIDIA Ada Lovelace（sm_89）架构上从零手写单精度矩阵乘法（SGEMM），通过 7 版迭代优化对标 cuBLAS。

这是个**学习项目**：每个 kernel 只做一项优化（共享内存 tile / 寄存器 tile / 向量化访存 / bank conflict 消除 / double buffering / 异步拷贝 cp.async），方便用 Nsight Compute metric 把每一步的性能影响单独归因。

v4 和 v5 是**两个没跑赢 v3 的优化**——在老架构上是经典 win，但在 Ada（sm_89）上反而变慢。把它们留在 repo 里，是因为失败的原因本身值得理解。v6 引入 `cp.async` 成为最快的 kernel——但它**为什么**快（靠 latency hiding，而不是你以为的 occupancy 回升）才是整个 repo 里最有意思的结果。

![N=4096 各版本 GFLOPS](docs/figures/performance_bars.png)

---

## 性能 Headline

| Kernel | GFLOPS @ M=N=K=4096 | 相对 Naive 加速 | 占 cuBLAS 百分比 |
|---|---:|---:|---:|
| v0 — Naive | 869 | 1.00× | 9.0% |
| v1 — 共享内存 tiling | 1,116 | 1.29× | 11.5% |
| v2 — 寄存器 tiling | 5,191 | 5.98× | 53.7% |
| v3 — + float4 向量化访存 | 7,683 | 8.85× | 79.5% |
| v4 — + 1-padding（反面案例）| 7,268 | 8.37× | 75.2% |
| v5 — Double buffering（反面案例）| 4,781 | 5.50× | 49.4% |
| **v6 — + cp.async pipeline** ⭐ | **8,708** | **10.02×** | **90.1%** |
| cuBLAS（FP32 SIMT, autotuned）| 9,668 | 11.13× | 100% |

> **GFLOPS** = 每秒十亿次浮点运算（Giga Floating-point Operations Per Second）。GEMM 的总 FLOPs = `2 × M × N × K`（每个 FMA 算 2 FLOPs：1 次乘 + 1 次加）。

最优手写 kernel（**v6, cp.async**）达到 **90% 的 cuBLAS 性能**——而这里的 cuBLAS *本身也是 FP32 SIMT*（9.7 TFLOPS ≈ FP32 峰值的 64%；若走 TF32 Tensor Core 会远超 ~15 TFLOPS 的 FP32 天花板，但我们没看到）。所以剩下的 ~10% 是**更好的 FP32-SIMT 工程**——shape autotuning、warp-level tiling、swizzle 布局、L2-aware 的 threadblock rasterization——**不是换了算力单元**。v6 自身的意外之处：它比 v5 快是靠 **latency hiding，而不是 occupancy**——见下方 v6 章节。

---

## 硬件环境

| 组件 | 规格 |
|---|---|
| GPU | NVIDIA RTX 4060 Laptop（Ada Lovelace, sm_89）|
| FP32 峰值 | ~15.1 TFLOPS |
| 显存带宽 | ~272 GB/s（GDDR6）|
| 共享内存 / SM | 100 KB unified L1 + shared（可配置）|
| 寄存器 / SM | 65,536 × 32-bit |
| 工具链 | CUDA 12.4, Nsight Compute 2024.1.1, WSL2 (Ubuntu) |

> 注：这里的 cuBLAS 调用是默认 math mode 下的 `cublasSgemm`。实测吞吐（~9.7 TFLOPS，≈ FP32 峰值的 64%）证明它在本机跑的是 **FP32-SIMT** kernel——若走 TF32 Tensor Core 会远高于 ~15 TFLOPS 的 FP32 天花板。TF32 tensor core *确实*可用（通过 `cublasMath` / `NVIDIA_TF32_OVERRIDE` 环境变量），是一个降精度的、更高的独立天花板——本项目不做。

---

## Kernel 版本说明

所有 kernel 共享相同的调用约定：`C = α · A·B + β · C`，行主序存储，支持任意 M / N / K。Block tile、K 维步长、Thread tile 大小都是模板参数；launch wrapper 选具体配置。

| 版本 | 文件 | 优化技术 | 结果 |
|---|---|---|---|
| v0 | `kernels/sgemm_v0_naive.cuh` | 1 thread 1 output；K 循环直接读 global memory。 | baseline |
| v1 | `kernels/sgemm_v1_tiling.cuh` | 共享内存 tile（TILE=32），block 内协作加载，经典 blocked matmul。 | 比 v0 快 30% |
| v2 | `kernels/sgemm_v2_register_tile.cuh` | 每 thread 算 8×8 输出 tile；BM=BN=128, BK=8；**标量** global → shared 加载。64 个寄存器累加器解锁 ILP。 | **比 v1 快 5.0×** |
| v3 | `kernels/sgemm_v3_vectorized.cuh` | v2 + `float4` 向量化 global → shared 加载（128-bit LDG）。load 指令数减半。 | **比 v2 快 1.36×**，float4 里程碑 |
| v4 | `kernels/sgemm_v4_bank_conflict_free.cuh` | v3 + `+1` 共享内存 padding，破坏 shared bank conflict 模式。 | **Ada 上 −3%**（反面案例）|
| v5 | `kernels/sgemm_v5_double_buffer.cuh` | v3 + 双缓冲共享 tile，让 global load 和 compute 并行。 | **Ada 上 −37%**（反面案例）|
| v6 | `kernels/sgemm_v6_cpasync.cuh` | v3 计算核心 + `cp.async`（`cuda::memcpy_async` + thread-scope `cuda::pipeline`）替换 global→shared 加载路径；2-stage 双缓冲。 | **比 v3 快 13%**，最优结果（90% cuBLAS）|

### v4 在 Ada 上为什么退化

`+1` padding 把 shared 内存的行 stride 从 16 字节对齐（8 / 128 floats）变成不对齐（9 / 129 floats）。原本的 `float4` 共享内存**写入**操作必须退化成 4 次标量写入，LSU pipe 上的指令数翻 4 倍。在 Ada（sm_89）上，节省下来的 bank-conflict 消耗小于增加的 store 指令开销，**净退化**。

在更老的架构（Volta / Turing）上 shared bank arbitration 开销更大，同样的优化会带来正向收益。**架构相关的优化**——这是这个反面案例最有教学价值的一课。

### v5 在 Ada 上为什么退化

软件 double buffering 让单 block 共享内存翻倍（8 KB → 16 KB），同 SM 能同时驻留的 block 数砍半，**occupancy 减半**。在没有硬件 async copy（`cp.async`，sm_80+）的情况下，nvcc 能调度的"加载/计算重叠"非常有限。Occupancy 损失大于 latency hiding 的收益。

后续路径是 `cp.async` + 多 stage pipeline——**已实现为下方的 v6**，而且结果推翻了"靠 occupancy"的假设。

### v6（cp.async）为什么快——以及为什么原因出人意料

v6 完全不动 v3 的计算核心，只把 global→shared 的加载路径换成 `cuda::memcpy_async`（在 sm_80+ 上 lower 成 `cp.async` / LDGSTS），由 thread-scope 的 `cuda::pipeline` + 2-stage 缓冲驱动。

最初的假设是：cp.async 能**把 v5 丢掉的 occupancy 救回来**，因为它去掉了驻留寄存器的预取暂存。Nsight Compute 说**并不是**这么回事：

| 指标 @ 4096 | v3 | v5 | v6 |
|---|---:|---:|---:|
| 寄存器 / 线程 | 128 | 149 | 157 |
| achieved occupancy | 33.0% | 16.7% | **16.6%** |
| shared bank conflict | 268 M | 268 M | 268 M |
| `long_scoreboard` stall | 1.66 | 1.88 | **0.075** |

v6 的 occupancy 和 v5 **完全一样**（16.6%），寄存器数甚至更高——occupancy 论点彻底失败。真正起作用的是 **latency hiding**：cp.async 把 global→shared 传输和计算重叠，把 `long_scoreboard`（等 global memory）stall 从 1.88 压到约 0.075。

因为 v5 和 v6 的 **occupancy 相同、bank conflict 也相同**，v5 → v6 的 +82% 几乎可以完全归因于这个重叠。教训是：`cp.async` 提供**两个独立的好处——register-bypass（→ occupancy）和 async overlap（→ latency hiding）**——在这个 kernel 上只有第二个兑现了。当延迟被另一种方式藏住时，低 occupancy 并不致命。瓶颈现在从 global memory 延迟转移到了 **shared memory bank conflict**（仍是 268 M）——这是未来 swizzle 布局 v7 的目标。

---

### 不同矩阵尺寸下的 scaling

![GFLOPS 随矩阵尺寸变化](docs/figures/performance_lines.png)

---

## 编译 & 运行

```bash
make                    # 编译 ./benchmark
./benchmark             # 跑全部尺寸，打印对比表
make clean              # 清理构建产物
```

要求：CUDA toolkit（≥ 11.8），`nvcc` 和 `cublas` 在 linker path 上。Makefile 默认 `sm_89`，其他 GPU 改 `ARCH`。

### 性能分析

每个版本在 N=4096 的 `.ncu-rep` 报告归档在 `nsight_reports/`：

```bash
# CLI section view
ncu --import nsight_reports/v3_vectorized_4096.ncu-rep --section SpeedOfLight
ncu --import nsight_reports/v3_vectorized_4096.ncu-rep --section MemoryWorkloadAnalysis
ncu --import nsight_reports/v3_vectorized_4096.ncu-rep --section WarpStateStats
```

要重新生成报告：

```bash
ncu --set full --kernel-name regex:sgemm_v3_vectorized \
    --launch-skip 39 --launch-count 1 \
    -o nsight_reports/v3_vectorized_4096 -f \
    ./benchmark
```

`--launch-skip 39 --launch-count 1` 跳过小尺寸的 warmup launch，profile M=N=K=4096 的第一发（稳态尺寸）。

GUI 查看：用 Windows / Linux 桌面的 Nsight Compute 直接打开 `.ncu-rep` 文件。

---

## 实验方法论

- **正确性参考**：cuBLAS `cublasSgemm` 是 ground truth。因为 cuBLAS 假设列主序，wrapper 实际计算 `Cᵀ = Bᵀ × Aᵀ`（参数置换调用），等价于把行主序的 `C` 直接得出来——**不需要任何显式转置操作**（同一段内存重新解读 = 自动转置）。
- **计时**：`cudaEvent_t` start/stop 对（GPU-side 计时，没有 host launch overhead）。3 次 untimed warmup + 10 次 timed iteration，取平均值。
- **正确性验证**：每个 kernel 输出和 cuBLAS 元素级对比，`max_abs_diff > 1e-2` 报警（loose tolerance 因为 `--use_fast_math` 启用）。
- **GFLOPS 公式**：`2 · M · N · K / time_seconds / 1e9`——每个 FMA 算 2 FLOPs。

---

## 仓库结构

```
.
├── benchmark.cu                              # benchmark 入口
├── kernels/
│   ├── sgemm_v0_naive.cuh
│   ├── sgemm_v1_tiling.cuh
│   ├── sgemm_v2_register_tile.cuh
│   ├── sgemm_v3_vectorized.cuh
│   ├── sgemm_v4_bank_conflict_free.cuh
│   ├── sgemm_v5_double_buffer.cuh
│   └── sgemm_v6_cpasync.cuh
├── tests/
│   └── test_v6_correctness.cu                # v6 正确性对账 cuBLAS
├── include/
│   └── common.cuh                            # CUDA_CHECK / CUBLAS_CHECK / GpuTimer / gflops()
├── nsight_reports/                           # 每版 .ncu-rep 归档
├── Makefile
├── LICENSE
└── README.md / README.zh.md
```

---

## 不做的部分

知道但没做：

- **Tensor Core / WMMA / `mma.sync`**——一个降精度（TF32/FP16）的、更高的独立天花板。**不是** cuBLAS 在这里超过 v6 的原因：本 benchmark 的 cuBLAS 是 FP32 SIMT（见硬件注）。手写 tensor-core kernel 是另一个独立项目。
- **Swizzle 共享布局（v7）**——用 XOR 置换列地址消除现在成为 v6 头号瓶颈的 268 M shared-bank conflict。下一步计划。
- **Warp specialization**（一个 CTA 内拆 producer/consumer）。
- **模板化 autotuning**——遍历 (BM, BN, BK, TM, TN) 空间针对 (M, N, K, sm_xx) 的编译/运行时搜索。

---

## 参考资料

- V. Volkov, *"Better Performance at Lower Occupancy"*, GTC 2010 —— ILP 替代 occupancy 的经典论文。
- NVIDIA CUDA C++ Programming Guide —— 内存模型、bank conflicts、向量化访存。
- NVIDIA Nsight Compute Documentation —— metric 定义参考（`sm__throughput`, `smsp__warp_issue_stalled_*`, `l1tex__t_bytes_*`）。
- CUTLASS（`github.com/NVIDIA/cutlass`）—— 工业级 hierarchical tiling / swizzling / async pipeline 实现参考。
- Lei Mao 的 CUDA 博客（`leimao.github.io`）—— 易读的 step-by-step GEMM 优化教程。

---

## License

MIT —— 见 [LICENSE](LICENSE)。
