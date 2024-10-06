#include <stdio.h>

// 声明设备函数
__device__ int add(int a, int b) {
    return a + b;
}

// 声明函数指针类型
typedef int (*FuncType)(int, int);

// CUDA kernel，接收一个函数指针
__global__ void kernel(FuncType func, int a, int b, int* result) {
    *result = (*func)(a, b);
}

int main() {
    // 定义并初始化变量
    int a = 5, b = 3;
    int result;

    // 在设备端分配内存以存储结果
    int* d_result;
    cudaMalloc(&d_result, sizeof(int));

    // 获取设备上函数指针的地址
    FuncType d_func;
    cudaMemcpyFromSymbol(&d_func, add, sizeof(FuncType));
    printf("Start\n");

    // 启动 kernel，传递函数指针和其他参数
    kernel<<<1, 1>>>(d_func, a, b, d_result);

    // 从设备端拷贝结果回主机端
    cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    // 输出结果
    printf("Result: %d\n", result);

    // 释放设备内存
    cudaFree(d_result);

    return 0;
}
