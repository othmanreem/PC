// pcc_par.cu - CUDA Pearson Correlation Coefficient
// Parallelizes mean computation, standard deviation, and correlation calculation on GPU

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <iostream>
#include <iomanip>
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

int COLS = 128;
int ROWS = 128;

/**
 * Generate matrix in-place using seeded drand48() for reproducible data.
 * Must match the sequential version exactly.
 **/
void generatematrix(double *matrix, unsigned long seed) {
    srand48((long)seed);
    for (int i = 0; i < ROWS * COLS; i++) {
        matrix[i] = drand48();
    }
}

// ============================================================================
// CUDA Kernels
// ============================================================================

/**
 * Kernel 1: Calculate row means
 * Each block handles one row, uses parallel reduction
 */
__global__ void calcMeanKernel(const double* matrix, double* mean, int rows, int cols) {
    extern __shared__ double sdata[];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    if (row >= rows) return;

    // Each thread sums elements strided by blockDim.x
    double sum = 0.0;
    for (int j = tid; j < cols; j += blockSize) {
        sum += matrix[row * cols + j];
    }
    sdata[tid] = sum;
    __syncthreads();

    // Parallel reduction in shared memory
    for (int s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        mean[row] = sdata[0] / (double)cols;
    }
}

/**
 * Kernel 2: Calculate (matrix - mean) and standard deviation for each row
 * Each block handles one row
 */
__global__ void calcMinusMeanStdKernel(const double* matrix, const double* mean,
                                        double* mm, double* std_dev, int rows, int cols) {
    extern __shared__ double sdata[];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    if (row >= rows) return;

    double row_mean = mean[row];
    double sum_sq = 0.0;

    // Each thread processes elements strided by blockDim.x
    for (int j = tid; j < cols; j += blockSize) {
        double diff = matrix[row * cols + j] - row_mean;
        mm[row * cols + j] = diff;
        sum_sq += diff * diff;
    }
    sdata[tid] = sum_sq;
    __syncthreads();

    // Parallel reduction for sum of squares
    for (int s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        std_dev[row] = sqrt(sdata[0]);
    }
}

/**
 * Kernel 3: Compute Pearson correlation for all pairs
 * Each thread computes one correlation coefficient
 * Uses the upper triangular indexing: pair_idx maps to (sample1, sample2)
 */
__global__ void pearsonKernel(const double* mm, const double* std_dev,
                               double* output, int rows, int cols, long long numPairs) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= numPairs) return;

    // Convert linear index to (sample1, sample2) pair
    // Upper triangular: sample1 < sample2
    // idx = sample1*(rows - 1) - sample1*(sample1-1)/2 + (sample2 - sample1 - 1)
    // Solve for sample1, sample2 from idx (using quadratic formula approximation)

    // Fast inverse mapping using approximation
    // sample1 ≈ rows - 2 - floor(sqrt(2*(numPairs - 1 - idx) + 0.25) - 0.5)
    long long temp = numPairs - 1 - idx;
    int sample1 = (int)(rows - 2 - (int)floor(sqrt(2.0 * temp + 0.25) - 0.5));
    if (sample1 < 0) sample1 = 0;

    // Calculate the starting index for sample1
    long long start_idx = (long long)sample1 * (rows - 1) - (long long)sample1 * (sample1 - 1) / 2;
    
    // Adjust sample1 if needed
    while (start_idx > idx && sample1 > 0) {
        sample1--;
        start_idx = (long long)sample1 * (rows - 1) - (long long)sample1 * (sample1 - 1) / 2;
    }
    while (sample1 < rows - 1) {
        long long next_start = (long long)(sample1 + 1) * (rows - 1) - (long long)(sample1 + 1) * sample1 / 2;
        if (next_start > idx) break;
        sample1++;
        start_idx = next_start;
    }

    int sample2 = sample1 + 1 + (int)(idx - start_idx);

    if (sample1 >= rows - 1 || sample2 >= rows) return;

    // Compute correlation
    double sum = 0.0;
    for (int i = 0; i < cols; i++) {
        sum += mm[sample1 * cols + i] * mm[sample2 * cols + i];
    }
    double r = sum / (std_dev[sample1] * std_dev[sample2]);

    // Output index: same as sequential (compact upper triangular)
    // Sequential stores at: sample1 * ROWS + sample2 - summ, where summ = sum(l=0..sample1+1)
    // summ = (sample1+2)*(sample1+1)/2
    long long summ = ((long long)(sample1 + 2) * (sample1 + 1)) / 2;
    long long out_idx = (long long)sample1 * rows + sample2 - summ;
    output[out_idx] = r;
}

/**
 * Alternative optimized Pearson kernel using shared memory
 * Each block handles multiple pairs, loading row data into shared memory
 */
__global__ void pearsonKernelOptimized(const double* mm, const double* std_dev,
                                        double* output, int rows, int cols, long long numPairs) {
    extern __shared__ double shared[];

    long long pairIdx = (long long)blockIdx.x;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    if (pairIdx >= numPairs) return;

    // Convert linear index to (sample1, sample2) pair
    long long temp = numPairs - 1 - pairIdx;
    int sample1 = (int)(rows - 2 - (int)floor(sqrt(2.0 * temp + 0.25) - 0.5));
    if (sample1 < 0) sample1 = 0;

    long long start_idx = (long long)sample1 * (rows - 1) - (long long)sample1 * (sample1 - 1) / 2;
    
    while (start_idx > pairIdx && sample1 > 0) {
        sample1--;
        start_idx = (long long)sample1 * (rows - 1) - (long long)sample1 * (sample1 - 1) / 2;
    }
    while (sample1 < rows - 1) {
        long long next_start = (long long)(sample1 + 1) * (rows - 1) - (long long)(sample1 + 1) * sample1 / 2;
        if (next_start > pairIdx) break;
        sample1++;
        start_idx = next_start;
    }

    int sample2 = sample1 + 1 + (int)(pairIdx - start_idx);

    if (sample1 >= rows - 1 || sample2 >= rows) return;

    // Parallel dot product using threads within this block
    double localSum = 0.0;
    for (int i = tid; i < cols; i += blockSize) {
        localSum += mm[sample1 * cols + i] * mm[sample2 * cols + i];
    }
    shared[tid] = localSum;
    __syncthreads();

    // Reduction
    for (int s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        double r = shared[0] / (std_dev[sample1] * std_dev[sample2]);
        long long summ = ((long long)(sample1 + 2) * (sample1 + 1)) / 2;
        long long out_idx = (long long)sample1 * rows + sample2 - summ;
        output[out_idx] = r;
    }
}

// ============================================================================
// Main PCC function
// ============================================================================

void pearson_cuda(double* h_matrix, double* h_output, long long cor_size) {
    // Device pointers
    double *d_matrix, *d_mean, *d_mm, *d_std, *d_output;

    size_t matrixSize = (size_t)ROWS * COLS * sizeof(double);
    size_t rowSize = ROWS * sizeof(double);
    size_t outputSize = cor_size * sizeof(double);

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_matrix, matrixSize));
    CUDA_CHECK(cudaMalloc(&d_mean, rowSize));
    CUDA_CHECK(cudaMalloc(&d_mm, matrixSize));
    CUDA_CHECK(cudaMalloc(&d_std, rowSize));
    CUDA_CHECK(cudaMalloc(&d_output, outputSize));

    // Copy input matrix to device
    CUDA_CHECK(cudaMemcpy(d_matrix, h_matrix, matrixSize, cudaMemcpyHostToDevice));

    // Determine kernel launch parameters
    int threadsPerBlock = 256;
    if (COLS < threadsPerBlock) {
        // Round up to next power of 2 >= COLS for efficient reduction
        threadsPerBlock = 1;
        while (threadsPerBlock < COLS) threadsPerBlock *= 2;
        if (threadsPerBlock > 1024) threadsPerBlock = 1024;
    }

    size_t sharedMemSize = threadsPerBlock * sizeof(double);

    // Step 1: Calculate row means
    calcMeanKernel<<<ROWS, threadsPerBlock, sharedMemSize>>>(d_matrix, d_mean, ROWS, COLS);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 2: Calculate (matrix - mean) and standard deviations
    calcMinusMeanStdKernel<<<ROWS, threadsPerBlock, sharedMemSize>>>(d_matrix, d_mean, d_mm, d_std, ROWS, COLS);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 3: Compute Pearson correlations
    long long numPairs = cor_size;

    // Choose between simple kernel (many threads per pair) or optimized (one block per pair)
    if (COLS <= 256) {
        // Simple kernel: each thread computes one pair
        int pairThreads = 256;
        int pairBlocks = (int)((numPairs + pairThreads - 1) / pairThreads);
        pearsonKernel<<<pairBlocks, pairThreads>>>(d_mm, d_std, d_output, ROWS, COLS, numPairs);
    } else {
        // Optimized kernel: each block computes one pair with parallel reduction
        int redThreads = 256;
        pearsonKernelOptimized<<<numPairs, redThreads, redThreads * sizeof(double)>>>(d_mm, d_std, d_output, ROWS, COLS, numPairs);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_output, d_output, outputSize, cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(d_matrix));
    CUDA_CHECK(cudaFree(d_mean));
    CUDA_CHECK(cudaFree(d_mm));
    CUDA_CHECK(cudaFree(d_std));
    CUDA_CHECK(cudaFree(d_output));
}

void writeoutput(double *output, long long cor_size, const char *name) {
    FILE *f = fopen(name, "wb");
    if (!f) {
        std::cerr << "Failed to open output file: " << name << std::endl;
        exit(1);
    }
    for (long long i = 0; i < cor_size; i++) {
        std::fprintf(f, "%.15g\n", output[i]);
    }
    fclose(f);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s matrix_height matrix_width [seed]\n", argv[0]);
        std::exit(-1);
    }

    ROWS = atoi(argv[1]);
    if (ROWS < 1) {
        std::fprintf(stderr, "error: height must be at least 1\n");
        std::exit(-1);
    }

    COLS = atoi(argv[2]);
    if (COLS < 1) {
        std::fprintf(stderr, "error: width must be at least 1\n");
        std::exit(-1);
    }

    unsigned long seed = 12345;
    if (argc >= 4) {
        seed = (unsigned long)atol(argv[3]);
    }

    // Print CUDA device info
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cerr << "GPU: " << prop.name << std::endl;
    }

    // Output filename (same format as sequential)
    char output_filename[30];
    snprintf(output_filename, 30, "pccout_%d_%d.dat", ROWS, COLS);

    // Calculate output size
    long long cor_size = (long long)(ROWS - 1) * ROWS / 2;

    // Allocate host memory
    double *matrix = (double*)malloc(sizeof(double) * COLS * ROWS);
    double *output = (double*)malloc(sizeof(double) * cor_size);

    if (matrix == NULL || output == NULL) {
        std::cerr << "Failed to allocate host memory" << std::endl;
        return 1;
    }

    // Generate matrix (same as sequential)
    generatematrix(matrix, seed);

    // Time the CUDA computation
    auto start = std::chrono::steady_clock::now();
    pearson_cuda(matrix, output, cor_size);
    auto end = std::chrono::steady_clock::now();

    std::cout << "Elapsed time =  " << std::fixed << std::setprecision(4)
              << std::chrono::duration<double>(end - start).count() << " sec\n";

    // Write output
    writeoutput(output, cor_size, output_filename);

    free(matrix);
    free(output);

    return 0;
}
