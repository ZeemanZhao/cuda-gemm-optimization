# RTX 4060 上的 CUDA GEMM 优化

[English](README.md) | **中文**

在 NVIDIA Ada Lovelace（sm_89）架构上从零手写单精度矩阵乘法（SGEMM），通过 7 版迭代优化对标 cuBLAS。

每个 kernel 在前一版基础上只做一项优化——共享内存 tile、寄存器 tile、向量化访存、bank conflict 处理、double buffering、异步拷贝——这样 Nsight Compute 能把每一步的影响单独归因。

v4 和 v5 在 Ada（sm_89）上反而变慢——尽管它们在老架构上是经典优化；作为 documented 反面案例保留。v6 引入 `cp.async`，v7 再加一层 XOR-swizzle 的共享布局把 shared bank conflict 降到零。最优版本 v7 在纯 FP32 SIMT 下达到 **~98% 的 cuBLAS**。

![N=4096 各版本 GFLOPS](docs/figures/performance_bars.png)

---

## 性能 Headline

| Kernel | GFLOPS @ M=N=K=4096 | 相对 Naive 加速 | 占 cuBLAS 百分比 |
|---|---:|---:|---:|
| v0 — Naive | 859 | 1.00× | 8.8% |
| v1 — 共享内存 tiling | 1,107 | 1.29× | 11.4% |
| v2 — 寄存器 tiling | 5,049 | 5.88× | 51.8% |
| v3 — + float4 向量化访存 | 7,685 | 8.95× | 78.9% |
| v4 — + 1-padding（反面案例）| 7,297 | 8.49× | 74.9% |
| v5 — Double buffering（反面案例）| 4,768 | 5.55× | 49.0% |
| v6 — + cp.async pipeline | 8,711 | 10.14× | 89.4% |
| **v7 — + Bs XOR-swizzle** ⭐ | **9,524** | **11.09×** | **97.8%** |
| cuBLAS（FP32 SIMT, autotuned）| 9,740 | 11.34× | 100% |

> **GFLOPS** = 每秒十亿次浮点运算（Giga Floating-point Operations Per Second）。GEMM 的总 FLOPs = `2 × M × N × K`（每个 FMA 算 2 FLOPs：1 次乘 + 1 次加）。

最优手写 kernel（**v7**）在纯 FP32 SIMT 下达到 **~98% 的 cuBLAS**。这里的 cuBLAS 本身也是 FP32 SIMT——~9.7 TFLOPS，约 FP32 峰值的 64%；若走 TF32 Tensor Core 会远超 ~15 TFLOPS 的 FP32 天花板。剩下的差距是 FP32-SIMT 工程（shape autotuning、occupancy），不是换了算力单元。下面各版本给出每一步的 Nsight Compute 归因。

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
| v6 | `kernels/sgemm_v6_cpasync.cuh` | v3 计算核心 + `cp.async`（`cuda::memcpy_async` + thread-scope `cuda::pipeline`）替换 global→shared 加载路径；2-stage 双缓冲。 | **比 v3 快 13%**（89% cuBLAS）|
| v7 | `kernels/sgemm_v7_swizzle.cuh` | v6 + 对 Bs 共享布局做 XOR swizzle（`perm(c4)=c4^(c4>>3)`，在 `cp.async` 写和 compute 读两处都用）消除 shared bank conflict。 | **比 v6 快 9%**，最优结果（~98% cuBLAS）|

### v4 在 Ada 上为什么退化

`+1` padding 把 shared 内存的行 stride 从 16 字节对齐（8 / 128 floats）变成不对齐（9 / 129 floats）。原本的 `float4` 共享内存**写入**操作必须退化成 4 次标量写入，LSU pipe 上的指令数翻 4 倍。在 Ada（sm_89）上，节省下来的 bank-conflict 消耗小于增加的 store 指令开销，**净退化**。

在更老的架构（Volta / Turing）上 shared bank arbitration 开销更大，同样的优化会带来正向收益——这个优化是架构相关的。

### v5 在 Ada 上为什么退化

软件 double buffering 让单 block 共享内存翻倍（8 KB → 16 KB），同 SM 能同时驻留的 block 数砍半，**occupancy 减半**。在没有硬件 async copy（`cp.async`，sm_80+）的情况下，nvcc 能调度的"加载/计算重叠"非常有限。Occupancy 损失大于 latency hiding 的收益。

后续路径是硬件 async（`cp.async` + 多 stage pipeline），实现为 **v6**。

### v6：cp.async —— latency hiding，不是 occupancy

v6 不动 v3 的计算核心，只把 global→shared 的加载路径换成 `cuda::memcpy_async`（在 sm_80+ 上 lower 成 `cp.async` / LDGSTS），由 thread-scope 的 `cuda::pipeline` + 2-stage 缓冲驱动。

`cp.async` 可能从两方面帮忙：绕过寄存器文件（或许能救回 v5 丢的 occupancy），以及让传输与计算重叠。Nsight Compute 显示这里只有第二点成立：

| 指标 @ 4096 | v3 | v5 | v6 |
|---|---:|---:|---:|
| 寄存器 / 线程 | 128 | 149 | 157 |
| achieved occupancy | 33.0% | 16.7% | **16.6%** |
| shared bank conflict | 268 M | 268 M | 268 M |
| `long_scoreboard` stall | 1.66 | 1.88 | **0.075** |

occupancy 和 v5 完全一样（16.6%），寄存器还略高：cp.async 在这里没救回 occupancy。提升完全来自 **latency hiding**——把 global→shared 传输与计算重叠，使 `long_scoreboard`（等 global memory）stall 从 1.88 降到约 0.075。

v5 与 v6 的 occupancy 相同、bank conflict 也相同，所以 v5 → v6 的 +82% 可归因于这个重叠。`cp.async` 提供两个独立的好处——register-bypass（→ occupancy）与 async overlap（→ latency hiding）——这个 kernel 上只有第二个兑现。瓶颈随之从 global memory 延迟转移到 **shared memory bank conflict**（仍是 268 M），由 v7 解决。

### v7：XOR swizzle 消除 bank conflict

v7 完全保留 v6，只改 Bs 的每一列**存在 shared 的哪个物理位置**。Bs 读是 4-way bank conflict：一个 warp 的 16 个 `float4` 列只落在 8 个 `float4` bank-组里的 4 个。XOR swizzle——`perm(c4) = c4 ^ (c4 >> 3)`，在 `cp.async` 写 **和** compute 读两处都用——把它们摊到全 8 组。它按整个 `float4` 置换，所以 16 字节对齐（以及 `LDS.128` 向量化）都保住了；v4 的 `+1` padding 做不到这点。

| 指标 @ 4096 | v6 | v7 |
|---|---:|---:|
| shared bank conflict | 268 M | **0** |
| shared-load wavefronts | 671 M | 384 M |
| `short_scoreboard` stall | 0.39 | 0.275 |
| GFLOPS | 8,711 | **9,524**（≈98% cuBLAS）|

shared bank conflict 降到 **零**，shared 访存流量降 43%；occupancy 不变（16.6%）。~98% of（FP32-SIMT）cuBLAS 已接近本设备手写 FP32 kernel 的实际天花板，剩余差距受 occupancy/调优制约。

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
│   ├── sgemm_v6_cpasync.cuh
│   └── sgemm_v7_swizzle.cuh
├── tests/
│   └── test_v6_correctness.cu                # v6 + v7 正确性对账 cuBLAS
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

- **Tensor Core / WMMA / `mma.sync`**——一个降精度（TF32/FP16）的、更高的独立天花板。**不是** cuBLAS 在这里(微弱)领先 v7 的原因：本 benchmark 的 cuBLAS 是 FP32 SIMT（见硬件注）。手写 tensor-core kernel 是另一个独立项目。
- **Warp-level 重排**——重排 warp 的 thread 布局（warp tiling）把 occupancy 从现在的 16.6% 抬上去；v7 之后到 cuBLAS 的剩余差距是 occupancy/调优制约，不再是 bank conflict。
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
