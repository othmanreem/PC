// oddevensort_par.cu
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

// Single-block kernel 
__global__ void oddEvenSort_SingleBlock(int *arr, int n) {
    int tid = threadIdx.x;
    for (int phase = 0; phase < n; phase++) {
        int i = 2 * tid + (phase % 2);
        if (i + 1 < n) {
            if (arr[i] > arr[i + 1]) {
                int tmp = arr[i];
                arr[i] = arr[i + 1];
                arr[i + 1] = tmp;
            }
        }
        __syncthreads(); // sync all threads after every phase
    }
}

// Multi-block kernel 

__global__ void oddEvenSort_OnePhase(int *arr, int n, int phase) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = 2 * tid + (phase % 2);
    if (i + 1 < n) {
        if (arr[i] > arr[i + 1]) {
            int tmp = arr[i];
            arr[i] = arr[i + 1];
            arr[i + 1] = tmp;
        }
    }
}

int isSorted(int *arr, int n) {
    for (int i = 0; i < n - 1; i++) {
        if (arr[i] > arr[i + 1]) return 0;
    }
    return 1;
}

int main(int argc, char** argv) {
     int n = 2048; 
    if (argc > 1) n = atoi(argv[1]);

    size_t size = n * sizeof(int);

    // Generate random input on Host 
    int *h_arr = (int *)malloc(size);
    srand(42);
    for (int i = 0; i < n; i++) h_arr[i] = rand() % 10000;

    // Allocate Device Memory 
    int *d_arr;
    cudaMalloc(&d_arr, size);

    // Timing Setup 
    float ms_single = 0, ms_multi = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // RUN 1: Single block version
    cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice);
    int threads_needed = n / 2; 

    if (threads_needed > 1024) {
        printf("Skipping Single-Block: n is too large (max 2048).\n");
    } else {
        cudaEventRecord(start);
        oddEvenSort_SingleBlock<<<1, threads_needed>>>(d_arr, n);
        cudaEventRecord(stop);
        cudaDeviceSynchronize();
        cudaEventElapsedTime(&ms_single, start, stop);

        int *res1 = (int *)malloc(size);
        cudaMemcpy(res1, d_arr, size, cudaMemcpyDeviceToHost);
        printf("Single-block | Sorted: %s | Time: %f ms\n", isSorted(res1, n) ? "YES" : "NO", ms_single);
        free(res1);
    }

    // RUN 2: Multi block version 
    cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice); // reset data

    int threadsPerBlock = 256;
    int blocks = (threads_needed + threadsPerBlock - 1) / threadsPerBlock;

    cudaEventRecord(start);
    for (int phase = 0; phase < n; phase++) {
        oddEvenSort_OnePhase<<<blocks, threadsPerBlock>>>(d_arr, n, phase);
    }
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&ms_multi, start, stop);

    int *res2 = (int *)malloc(size);
    cudaMemcpy(res2, d_arr, size, cudaMemcpyDeviceToHost);
    printf("Multi-block  | Sorted: %s | Time: %f ms\n", isSorted(res2, n) ? "YES" : "NO", ms_multi);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_arr);
    free(h_arr); 
    free(res2);
    return 0;
}