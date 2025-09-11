#include "main.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <climits>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file>" << std::endl;
        return 1;
    }
    
    std::ifstream input_file(argv[1]);
    if (!input_file.is_open()) {
        std::cerr << "Error: Cannot open file " << argv[1] << std::endl;
        return 1;
    }
    
    int V, E;
    input_file >> V >> E;
    
    // 读取边数据
    std::vector<int> edges;
    edges.reserve(E * 3);
    for (int i = 0; i < E; i++) {
        int src, dst, weight;
        input_file >> src >> dst >> weight;
        edges.push_back(src);
        edges.push_back(dst);
        edges.push_back(weight);
    }
    input_file.close();
    
    // 分配结果矩阵内存
    std::vector<int> result(V * V);
    
    // 调用GPU求解函数
    apsp(V, E, edges.data(), result.data());
    
    // 输出结果（按行优先顺序）
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
            std::cout << result[i * V + j] << " ";
        }std::cout << std::endl;
    }
    
    
    return 0;
}