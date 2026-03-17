// oddevensort_par.cu

#include <vector>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " code=" << (int)err << " (" << cudaGetErrorString(err) << ")" << std::endl; \
            exit(1); \
        } \
    } while(0)

// Single-block kernel 
// All threads live in ONE block. Phases are serialized by __syncthreads().
__global__ void oddeven_sort_single_block(int* d_arr, int n) {
    int tid = threadIdx.x;

    for (int phase = 1; phase <= n; phase++) {
        int start = phase % 2; // 0 = even phase, 1 = odd phase

        // Each thread handles one pair: index (start + 2*tid)
        int j = start + 2 * tid;
        if (j < n - 1) {
            if (d_arr[j] > d_arr[j + 1]) {
                int tmp = d_arr[j];
                d_arr[j] = d_arr[j + 1];
                d_arr[j + 1] = tmp;
            }
        }
        __syncthreads(); // Barrier between every phase
    }
}

__global__ void oddeven_phase_multi_block(int* d_arr, int n, int phase_parity) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int j = phase_parity + 2 * tid; // which pair this thread compares

    if (j < n - 1) {
        if (d_arr[j] > d_arr[j + 1]) {
            int tmp = d_arr[j];
            d_arr[j] = d_arr[j + 1];
            d_arr[j + 1] = tmp;
        }
    }
}

void run_multi_block(int* d_arr, int n, int threadsPerBlock) {
    int pairs = n / 2;
    int blocks = (pairs + threadsPerBlock - 1) / threadsPerBlock;

    for (int phase = 1; phase <= n; phase++) {
        int parity = phase % 2; // 0 or 1
        oddeven_phase_multi_block<<<blocks, threadsPerBlock>>>(d_arr, n, parity);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize()); // Global sync between phases
    }
}

int main() {
    // Check CUDA device
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    std::cout << "CUDA Device Count: " << deviceCount << std::endl;
    if (deviceCount > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "GPU: " << prop.name << std::endl;
    }
    
    const int SIZE = 1 << 19; 
    const int THREADS = 512;

    // Generate random data
    std::vector<int> h_original(SIZE);
    srand(time(0));
    std::generate(h_original.begin(), h_original.end(), rand);

    // Allocate device memory
    int* d_arr;
    CUDA_CHECK(cudaMalloc(&d_arr, SIZE * sizeof(int)));

    // VARIANT 1: Single-block (limited to 1024*2 = 2048 elements)
    {
        const int SMALL = 2048; // Single-block limit
        std::vector<int> h_data(h_original.begin(), h_original.begin() + SMALL);

        CUDA_CHECK(cudaMemcpy(d_arr, h_data.data(), SMALL * sizeof(int), cudaMemcpyHostToDevice));

        auto start = std::chrono::steady_clock::now();
        oddeven_sort_single_block<<<1, SMALL / 2>>>(d_arr, SMALL);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::steady_clock::now();

        CUDA_CHECK(cudaMemcpy(h_data.data(), d_arr, SMALL * sizeof(int), cudaMemcpyDeviceToHost));

        bool ok = std::is_sorted(h_data.begin(), h_data.end());
        std::cout << "[Single-block] Sorted: " << (ok ? "YES" : "NO") << "\n";
        std::cout << "[Single-block] Time: "
                  << std::chrono::duration<double>(end - start).count() << " s\n\n";
    }

    // VARIANT 2: Multi-block (scales to arbitrary sizes)
    {
        std::vector<int> h_data = h_original; // Full 2^19 array

        CUDA_CHECK(cudaMemcpy(d_arr, h_data.data(), SIZE * sizeof(int), cudaMemcpyHostToDevice));

        auto start = std::chrono::steady_clock::now();
        run_multi_block(d_arr, SIZE, THREADS);
        auto end = std::chrono::steady_clock::now();

        CUDA_CHECK(cudaMemcpy(h_data.data(), d_arr, SIZE * sizeof(int), cudaMemcpyDeviceToHost));

        bool ok = std::is_sorted(h_data.begin(), h_data.end());
        std::cout << "[Multi-block]  Sorted: " << (ok ? "YES" : "NO") << "\n";
        std::cout << "[Multi-block]  Time: "
                  << std::chrono::duration<double>(end - start).count() << " s\n\n";
    }

    cudaFree(d_arr);
    return 0;
}

