#include "main.h"
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>

#define CHECK_HIP(cmd) { hipError_t error = cmd; if (error != hipSuccess) { \
    std::cerr << "HIP error: " << hipGetErrorString(error) << " at line " << __LINE__ << std::endl; \
    exit(EXIT_FAILURE); }}

const int BLOCK_SIZE = 256;
const int ELEMENTS_PER_THREAD = 4;

// 使用共享内存的块内扫描（Blelloch算法）
__device__ void block_scan(int* data) {
    extern __shared__ int temp[];
    int tid = threadIdx.x;
    
    // 上扫阶段（reduction）
    int offset = 1;
    for (int d = BLOCK_SIZE >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    
    // 清除最后一个元素
    if (tid == 0) {
        temp[BLOCK_SIZE - 1] = 0;
    }
    
    // 下扫阶段
    for (int d = 1; d < BLOCK_SIZE; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();
}

// 第一阶段：块内扫描并收集块总和
__global__ void phase1_kernel(const int* input, int* output, int* block_sums, int n) {
    extern __shared__ int shared_data[];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int block_start = bid * BLOCK_SIZE * ELEMENTS_PER_THREAD;
    
    // 每个线程加载多个元素并计算局部和
    int local_data[ELEMENTS_PER_THREAD];
    int local_sum = 0;
    
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        int idx = block_start + tid * ELEMENTS_PER_THREAD + i;
        local_data[i] = (idx < n) ? input[idx] : 0;
        local_sum += local_data[i];
    }
    
    // 将每个线程的和存入共享内存
    shared_data[tid] = local_sum;
    __syncthreads();
    
    // 在共享内存上进行扫描
    block_scan(shared_data);
    
    // 存储块总和
    if (tid == BLOCK_SIZE - 1) {
        block_sums[bid] = shared_data[tid] + local_sum;
    }
    
    // 计算并存储最终结果
    int block_prefix = shared_data[tid];
    int running_sum = block_prefix;
    
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        int idx = block_start + tid * ELEMENTS_PER_THREAD + i;
        if (idx < n) {
            running_sum += local_data[i];
            output[idx] = running_sum;
        }
    }
}

// 第二阶段：将块偏移量加到结果上
__global__ void phase2_kernel(int* output, const int* block_offsets, int n) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int block_start = bid * BLOCK_SIZE * ELEMENTS_PER_THREAD;
    
    int offset = block_offsets[bid];
    
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        int idx = block_start + tid * ELEMENTS_PER_THREAD + i;
        if (idx < n) {
            output[idx] += offset;
        }
    }
}

// 简单的CPU前缀和用于验证和小数组
void cpu_prefix_sum(const int* input, int* output, int n) {
    if (n == 0) return;
    output[0] = input[0];
    for (int i = 1; i < n; i++) {
        output[i] = output[i-1] + input[i];
    }
}

void solve(const int* input, int* output, int n) {
    if (n == 0) return;
    
    // 对于小数组，使用CPU版本更高效
    if (n <= 1024) {
        cpu_prefix_sum(input, output, n);
        return;
    }
    
    // 计算块数
    int elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    int num_blocks = (n + elements_per_block - 1) / elements_per_block;
    
    // 分配设备内存
    int *d_input, *d_output, *d_block_sums;
    
    CHECK_HIP(hipMalloc(&d_input, n * sizeof(int)));
    CHECK_HIP(hipMalloc(&d_output, n * sizeof(int)));
    CHECK_HIP(hipMalloc(&d_block_sums, num_blocks * sizeof(int)));
    
    // 拷贝输入数据
    CHECK_HIP(hipMemcpy(d_input, input, n * sizeof(int), hipMemcpyHostToDevice));
    
    // 第一阶段：块内扫描
    dim3 block_dim(BLOCK_SIZE);
    dim3 grid_dim(num_blocks);
    size_t shared_mem_size = BLOCK_SIZE * sizeof(int);
    
    hipLaunchKernelGGL(phase1_kernel, grid_dim, block_dim, shared_mem_size, 0,
                      d_input, d_output, d_block_sums, n);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());
    
    // 在CPU上计算块偏移量（排除式前缀和）
    std::vector<int> block_sums(num_blocks);
    std::vector<int> block_offsets(num_blocks);
    
    CHECK_HIP(hipMemcpy(block_sums.data(), d_block_sums, num_blocks * sizeof(int), hipMemcpyDeviceToHost));
    
    block_offsets[0] = 0;
    for (int i = 1; i < num_blocks; i++) {
        block_offsets[i] = block_offsets[i-1] + block_sums[i-1];
    }
    
    // 将块偏移量拷贝回设备
    int *d_block_offsets;
    CHECK_HIP(hipMalloc(&d_block_offsets, num_blocks * sizeof(int)));
    CHECK_HIP(hipMemcpy(d_block_offsets, block_offsets.data(), num_blocks * sizeof(int), hipMemcpyHostToDevice));
    
    // 第二阶段：添加块偏移量
    hipLaunchKernelGGL(phase2_kernel, grid_dim, block_dim, 0, 0,
                      d_output, d_block_offsets, n);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());
    
    // 拷贝结果回主机
    CHECK_HIP(hipMemcpy(output, d_output, n * sizeof(int), hipMemcpyDeviceToHost));
    
    // 清理
    CHECK_HIP(hipFree(d_input));
    CHECK_HIP(hipFree(d_output));
    CHECK_HIP(hipFree(d_block_sums));
    CHECK_HIP(hipFree(d_block_offsets));
}
