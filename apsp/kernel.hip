// kernel.hip
#include "main.h"
#include <hip/hip_runtime.h>
#include <algorithm>

#define INF (1 << 30) - 1
#define BLOCK_SIZE 32

// 初始化距离矩阵内核
__global__ void initDistanceMatrix(int V, int* dist) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < V && j < V) {
        int idx = i * V + j;
        if (i == j) {
            dist[idx] = 0;
        } else {
            dist[idx] = INF;
        }
    }
}

// 加载边数据到距离矩阵内核
__global__ void loadEdges(int V, int E, int* edges, int* dist) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < E) {
        int src = edges[3 * idx];
        int dst = edges[3 * idx + 1];
        int weight = edges[3 * idx + 2];
        dist[src * V + dst] = weight;
    }
}

// 分块Floyd-Warshall内核 - 更新枢轴块
__global__ void updatePivotBlock(int V, int B, int k, int* dist) {
    __shared__ int pivot_block[BLOCK_SIZE][BLOCK_SIZE];
    
    int i_local = threadIdx.x;
    int j_local = threadIdx.y;
    
    int i_global = k * B + i_local;
    int j_global = k * B + j_local;
    
    // 加载枢轴块到共享内存
    if (i_global < V && j_global < V) {
        pivot_block[i_local][j_local] = dist[i_global * V + j_global];
    } else {
        pivot_block[i_local][j_local] = INF;
    }
    __syncthreads();
    
    // 在共享内存中执行Floyd-Warshall
    for (int kk = 0; kk < B; kk++) {
        __syncthreads();
        int via_kk = INF;
        if (pivot_block[i_local][kk] != INF && pivot_block[kk][j_local] != INF) {
            via_kk = pivot_block[i_local][kk] + pivot_block[kk][j_local];
        }
        if (via_kk < pivot_block[i_local][j_local]) {
            pivot_block[i_local][j_local] = via_kk;
        }
    }
    __syncthreads();
    
    // 写回全局内存
    if (i_global < V && j_global < V) {
        dist[i_global * V + j_global] = pivot_block[i_local][j_local];
    }
}

// 分块Floyd-Warshall内核 - 更新行块和列块
__global__ void updateRowColBlocks(int V, int B, int k, int* dist, bool isRow) {
    __shared__ int pivot_block[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int target_block[BLOCK_SIZE][BLOCK_SIZE];
    
    int block_id = blockIdx.x;
    int i_local = threadIdx.x;
    int j_local = threadIdx.y;
    
    int i_global, j_global;
    if (isRow) {
        i_global = block_id * B + i_local;
        j_global = k * B + j_local;
    } else {
        i_global = k * B + i_local;
        j_global = block_id * B + j_local;
    }
    
    // 加载枢轴块和目标块到共享内存
    if (k * B + i_local < V && k * B + j_local < V) {
        pivot_block[i_local][j_local] = dist[(k * B + i_local) * V + (k * B + j_local)];
    } else {
        pivot_block[i_local][j_local] = INF;
    }
    
    if (i_global < V && j_global < V) {
        target_block[i_local][j_local] = dist[i_global * V + j_global];
    } else {
        target_block[i_local][j_local] = INF;
    }
    __syncthreads();
    
    // 更新目标块
    for (int kk = 0; kk < B; kk++) {
        __syncthreads();
        int via_kk = INF;
        if (isRow) {
            if (target_block[i_local][kk] != INF && pivot_block[kk][j_local] != INF) {
                via_kk = target_block[i_local][kk] + pivot_block[kk][j_local];
            }
        } else {
            if (pivot_block[i_local][kk] != INF && target_block[kk][j_local] != INF) {
                via_kk = pivot_block[i_local][kk] + target_block[kk][j_local];
            }
        }
        if (via_kk < target_block[i_local][j_local]) {
            target_block[i_local][j_local] = via_kk;
        }
    }
    __syncthreads();
    
    // 写回全局内存
    if (i_global < V && j_global < V) {
        dist[i_global * V + j_global] = target_block[i_local][j_local];
    }
}

// 分块Floyd-Warshall内核 - 更新剩余块
__global__ void updateOtherBlocks(int V, int B, int k, int* dist) {
    __shared__ int row_block[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int col_block[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int target_block[BLOCK_SIZE][BLOCK_SIZE];
    
    int i_block = blockIdx.x;
    int j_block = blockIdx.y;
    int i_local = threadIdx.x;
    int j_local = threadIdx.y;
    
    int i_global = i_block * B + i_local;
    int j_global = j_block * B + j_local;
    int k_global = k * B + threadIdx.x;
    
    // 加载行块、列块和目标块到共享内存
    if (i_global < V && k * B + j_local < V) {
        row_block[i_local][j_local] = dist[i_global * V + (k * B + j_local)];
    } else {
        row_block[i_local][j_local] = INF;
    }
    
    if (k * B + i_local < V && j_global < V) {
        col_block[i_local][j_local] = dist[(k * B + i_local) * V + j_global];
    } else {
        col_block[i_local][j_local] = INF;
    }
    
    if (i_global < V && j_global < V) {
        target_block[i_local][j_local] = dist[i_global * V + j_global];
    } else {
        target_block[i_local][j_local] = INF;
    }
    __syncthreads();
    
    // 更新目标块
    for (int kk = 0; kk < B; kk++) {
        __syncthreads();
        int via_kk = INF;
        if (row_block[i_local][kk] != INF && col_block[kk][j_local] != INF) {
            via_kk = row_block[i_local][kk] + col_block[kk][j_local];
        }
        if (via_kk < target_block[i_local][j_local]) {
            target_block[i_local][j_local] = via_kk;
        }
    }
    __syncthreads();
    
    // 写回全局内存
    if (i_global < V && j_global < V) {
        dist[i_global * V + j_global] = target_block[i_local][j_local];
    }
}

void apsp(int V, int E, int* edges, int* result) {
    int *d_edges, *d_dist;
    
    // 分配设备内存
    size_t dist_size = V * V * sizeof(int);
    size_t edges_size = E * 3 * sizeof(int);
    
    hipMalloc(&d_edges, edges_size);
    hipMalloc(&d_dist, dist_size);
    
    // 拷贝边数据到设备
    hipMemcpy(d_edges, edges, edges_size, hipMemcpyHostToDevice);
    
    // 初始化距离矩阵
    dim3 initBlocks((V + 15) / 16, (V + 15) / 16);
    dim3 initThreads(16, 16);
    initDistanceMatrix<<<initBlocks, initThreads>>>(V, d_dist);
    hipDeviceSynchronize();
    
    // 加载边数据
    int edgeThreads = 256;
    int edgeBlocks = (E + edgeThreads - 1) / edgeThreads;
    loadEdges<<<edgeBlocks, edgeThreads>>>(V, E, d_edges, d_dist);
    hipDeviceSynchronize();
    
    // 分块Floyd-Warshall算法
    int B = BLOCK_SIZE;
    int num_blocks = (V + B - 1) / B;
    
    dim3 blockDims(B, B);
    
    for (int k = 0; k < num_blocks; k++) {
        // 1. 更新枢轴块
        updatePivotBlock<<<1, blockDims>>>(V, B, k, d_dist);
        hipDeviceSynchronize();
        
        // 2. 更新行块
        dim3 rowColBlocks(num_blocks);
        updateRowColBlocks<<<rowColBlocks, blockDims>>>(V, B, k, d_dist, true);
        hipDeviceSynchronize();
        
        // 3. 更新列块
        updateRowColBlocks<<<rowColBlocks, blockDims>>>(V, B, k, d_dist, false);
        hipDeviceSynchronize();
        
        // 4. 更新剩余块
        dim3 otherBlocks(num_blocks, num_blocks);
        updateOtherBlocks<<<otherBlocks, blockDims>>>(V, B, k, d_dist);
        hipDeviceSynchronize();
    }
    
    // 拷贝结果回主机
    hipMemcpy(result, d_dist, dist_size, hipMemcpyDeviceToHost);
    
    // 释放设备内存
    hipFree(d_edges);
    hipFree(d_dist);
}
