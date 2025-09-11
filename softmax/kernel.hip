#include "main.h"
#include <cmath>
#define BLOCK_SIZE 256
#define MAX_BLOCKS 65535

// 寻找最大值的核函数
__global__ void findMaxKernel(const float* input, float* maxValue, int N) {
    __shared__ float shared_max[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    // 初始化共享内存
    shared_max[tid] = -INFINITY;
    
    // 每个线程处理多个元素
    float thread_max = -INFINITY;
    while (idx < N) {
        if (input[idx] > thread_max) {
            thread_max = input[idx];
        }
        idx += gridDim.x * blockDim.x;
    }
    
    shared_max[tid] = thread_max;
    __syncthreads();
    
    // 在块内规约求最大值
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (shared_max[tid + stride] > shared_max[tid]) {
                shared_max[tid] = shared_max[tid + stride];
            }
        }
        __syncthreads();
    }
    
    // 第一个线程将块最大值写入全局内存
    if (tid == 0) {
        maxValue[blockIdx.x] = shared_max[0];
    }
}

// 计算指数和的核函数
__global__ void computeExpSumKernel(const float* input, float* expSum, float maxVal, int N) {
    __shared__ float shared_sum[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    shared_sum[tid] = 0.0f;
    
    // 每个线程处理多个元素
    float thread_sum = 0.0f;
    while (idx < N) {
        float exp_val = expf(input[idx] - maxVal);
        thread_sum += exp_val;
        idx += gridDim.x * blockDim.x;
    }
    
    shared_sum[tid] = thread_sum;
    __syncthreads();
    
    // 在块内规约求和
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    
    // 第一个线程将块和写入全局内存
    if (tid == 0) {
        expSum[blockIdx.x] = shared_sum[0];
    }
}

// 计算softmax的核函数
__global__ void softmaxKernel(const float* input, float* output, float maxVal, float sum, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 每个线程处理多个元素
    while (idx < N) {
        float exp_val = expf(input[idx] - maxVal);
        output[idx] = exp_val / sum;
        idx += gridDim.x * blockDim.x;
    }
}


extern "C" void solve(const float* input, float* output, int N) {
    if (N == 0) return;
    
    // 计算需要的块数
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (numBlocks > MAX_BLOCKS) numBlocks = MAX_BLOCKS;
    
    // 分配设备内存
    float *d_input, *d_output;
    float *d_maxValues, *d_expSums;
    
    hipMalloc(&d_input, N * sizeof(float));
    hipMalloc(&d_output, N * sizeof(float));
    hipMalloc(&d_maxValues, numBlocks * sizeof(float));
    hipMalloc(&d_expSums, numBlocks * sizeof(float));
    
    // 拷贝输入数据到设备
    hipMemcpy(d_input, input, N * sizeof(float), hipMemcpyHostToDevice);
    
    // 第一步：找到全局最大值
    hipLaunchKernelGGL(findMaxKernel, dim3(numBlocks), dim3(BLOCK_SIZE), 0, 0, 
                      d_input, d_maxValues, N);
    
    // 将块最大值拷贝回主机并在主机上求全局最大值
    std::vector<float> maxValues(numBlocks);
    hipMemcpy(maxValues.data(), d_maxValues, numBlocks * sizeof(float), hipMemcpyDeviceToHost);
    
    float globalMax = -INFINITY;
    for (int i = 0; i < numBlocks; i++) {
        if (maxValues[i] > globalMax) {
            globalMax = maxValues[i];
        }
    }
    
    // 第二步：计算指数和
    hipLaunchKernelGGL(computeExpSumKernel, dim3(numBlocks), dim3(BLOCK_SIZE), 0, 0, 
                      d_input, d_expSums, globalMax, N);
    
    // 将块和拷贝回主机并在主机上求全局和
    std::vector<float> expSums(numBlocks);
    hipMemcpy(expSums.data(), d_expSums, numBlocks * sizeof(float), hipMemcpyDeviceToHost);
    
    float globalSum = 0.0f;
    for (int i = 0; i < numBlocks; i++) {
        globalSum += expSums[i];
    }
    
    // 防止除零错误
    if (globalSum < 1e-12f) globalSum = 1e-12f;
    
    // 第三步：计算softmax
    hipLaunchKernelGGL(softmaxKernel, dim3(numBlocks), dim3(BLOCK_SIZE), 0, 0, 
                      d_input, d_output, globalMax, globalSum, N);
    
    // 拷贝结果回主机
    hipMemcpy(output, d_output, N * sizeof(float), hipMemcpyDeviceToHost);
    
    // 释放设备内存
    hipFree(d_input);
    hipFree(d_output);
    hipFree(d_maxValues);
    hipFree(d_expSums);

}
