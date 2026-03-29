// oddevensort_par.cu

#include <vector>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

// handling error part
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " code=" << (int)err << " (" << cudaGetErrorString(err) << ")" << std::endl; \
            exit(1); \
        } \
    } while(0)

// Single-block version
__global__ void oddeven_sort_single_block(int* d_arr, int n) {
    int tid       = threadIdx.x;
    int blockSize = blockDim.x;   // always 1024

    for (int phase = 1; phase <= n; phase++) {
        int start = phase % 2;  // 1 = odd phase, 0 = even phase

        for (int j = start + 2 * tid; j < n - 1; j += 2 * blockSize) {
            if (d_arr[j] > d_arr[j + 1]) {
                int tmp        = d_arr[j];
                d_arr[j]       = d_arr[j + 1];
                d_arr[j + 1]   = tmp;
            }
        }

        // Barrier: no thread starts the next phase until every thread
        // has finished all its swaps in the current phase.
        __syncthreads();
    }
}


// each thread handles exactly one pair.
__global__ void oddeven_phase_multi_block(int* d_arr, int n, int phase_parity) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;  // unique id 
    int j = phase_parity + 2 * tid; // which pair this thread compares

    // kernel is repeated. Kernel is launched and does odd phase and then after it's finished it again gets launched and does even phase.
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

        // launch the kernel
        oddeven_phase_multi_block<<<blocks, threadsPerBlock>>>(d_arr, n, parity); 
        CUDA_CHECK(cudaGetLastError()); // This uses macro to check if the kernel launch failed.

        CUDA_CHECK(cudaDeviceSynchronize()); // Global sync between phase
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

    // setup 
    const int SIZE             = 1 << 19;  
    const int THREADS    = 512;    

    int* d_arr; 
    CUDA_CHECK(cudaMalloc(&d_arr, SIZE * sizeof(int))); 

    std::vector<int> h_orig_large(SIZE);
    srand(42);
    std::generate(h_orig_large.begin(), h_orig_large.end(), rand);

    // VARIANT 1: Single-block 
    {
        std::vector<int> h_data = h_orig_large;
        CUDA_CHECK(cudaMemcpy(d_arr, h_data.data(), SIZE * sizeof(int), cudaMemcpyHostToDevice));

        auto start = std::chrono::steady_clock::now();
        oddeven_sort_single_block<<<1, 1024>>>(d_arr, SIZE);
        CUDA_CHECK(cudaGetLastError()); 
        CUDA_CHECK(cudaDeviceSynchronize());  
        auto end = std::chrono::steady_clock::now();

        // Pulls the sorted data back from the GPU to the CPU
        CUDA_CHECK(cudaMemcpy(h_data.data(), d_arr, SIZE * sizeof(int), cudaMemcpyDeviceToHost));

        bool ok = std::is_sorted(h_data.begin(), h_data.end());
        std::cout << "[Single-block] n=" << SIZE
          << " Sorted=" << (ok ? "YES" : "NO")
          << " Time=" << std::chrono::duration<double>(end - start).count() << "s\n\n";
    }

    // VARIANT 2: Multi-block 
    for (int n : {100000, SIZE}) {
        std::vector<int> h_data(n);
    srand(42);
    std::generate(h_data.begin(), h_data.end(), rand);

    CUDA_CHECK(cudaMemcpy(d_arr, h_data.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    auto t0 = std::chrono::steady_clock::now();
    run_multi_block(d_arr, n, THREADS);
    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    CUDA_CHECK(cudaMemcpy(h_data.data(), d_arr, n * sizeof(int), cudaMemcpyDeviceToHost));

    bool ok = std::is_sorted(h_data.begin(), h_data.end());
    std::cout << "[Multi-block]  n=" << n
              << " Sorted=" << (ok ? "YES" : "NO")
              << " Time=" << elapsed << "s\n\n";
    }

    cudaFree(d_arr); // Releases the GPU memory we allocated at the start.
    return 0;
}

