NVCC     := nvcc
# sm_89 = Ada Lovelace (RTX 4060 Laptop)
ARCH     := -arch=sm_89
CFLAGS   := -O3 -std=c++17 $(ARCH) --use_fast_math
INCLUDES := -I.
LIBS     := -lcublas
TARGET   := benchmark

.PHONY: all clean run profile

all: $(TARGET)

$(TARGET): benchmark.cu \
           include/common.cuh \
           kernels/sgemm_naive.cuh \
           kernels/sgemm_tiling.cuh \
           kernels/sgemm_vectorized.cuh
	$(NVCC) $(CFLAGS) $(INCLUDES) $(LIBS) -o $@ benchmark.cu

run: $(TARGET)
	./$(TARGET)

# Optional: profile with Nsight Compute
# Requires: ncu (Nsight Compute CLI)
profile: $(TARGET)
	/opt/nvidia/nsight-compute/2024.1.1/ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second,\
sm__sass_thread_inst_executed_op_fadd_pred_on.sum,\
sm__sass_thread_inst_executed_op_fmul_pred_on.sum,\
sm__sass_thread_inst_executed_op_ffma_pred_on.sum \
	    --target-processes all \
	    ./$(TARGET)

clean:
	rm -f $(TARGET)
