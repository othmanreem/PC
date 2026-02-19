#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <omp.h>

int COLS = 128;
int ROWS = 128;

/**
 * Generate matrix in-place using seeded drand48() for reproducible data.
 */
void generatematrix(double *matrix, unsigned long seed)
{
    srand48((long)seed);
    for (int i = 0; i < ROWS * COLS; i++) {
        matrix[i] = drand48();
    }
}

/**
 * Calculate row mean — parallelised over rows.
 * Each row's sum is fully independent
 */
void calcmean(double *matrix, double *mean, int num_threads)
{
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    // Stating the compiler to split the next for loop across multiple threads using num_threads and divide the iterations evenly between threads 
    for (int i = 0; i < ROWS; i++) {
        double sum = 0.0;
        for (int j = 0; j < COLS; j++) {
            sum += matrix[i * COLS + j];
        }
        mean[i] = sum / (double)COLS;
    }
}

/**
 * Calculate matrix - rowmean, and standard deviation for every row 
 */
void calc_mm_std(double *matrix, double *mean, double *mm, double *std_dev, int num_threads)
{
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int i = 0; i < ROWS; i++) {
        double sum = 0.0;
        for (int j = 0; j < COLS; j++) {
            double diff = matrix[i * COLS + j] - mean[i];
            mm[i * COLS + j] = diff;
            sum += diff * diff;
        }
        std_dev[i] = std::sqrt(sum);
    }
}

void pearson(double *mm, double *std_dev, double *output, int num_threads)
{
    long long *offsets = (long long *)malloc(sizeof(long long) * ROWS);
    {
        long long acc = 0;
        for (int s1 = 0; s1 < ROWS - 1; s1++) {
            offsets[s1] = acc;
            acc += (ROWS - 1 - s1);
        }
    }

    #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 4)
    for (int sample1 = 0; sample1 < ROWS - 1; sample1++) {
        long long base = offsets[sample1];
        for (int sample2 = sample1 + 1; sample2 < ROWS; sample2++) {
            double sum = 0.0;
            for (int i = 0; i < COLS; i++) {
                sum += mm[sample1 * COLS + i] * mm[sample2 * COLS + i];
            }
            double r = sum / (std_dev[sample1] * std_dev[sample2]);
            output[base + (sample2 - sample1 - 1)] = r;
        }
    }

    free(offsets);
}

void pearson_par(double *input, double *output, int num_threads)
{
    double *mean     = (double *)malloc(sizeof(double) * ROWS);
    double *std_dev  = (double *)malloc(sizeof(double) * ROWS);
    double *mm       = (double *)malloc(sizeof(double) * ROWS * COLS);

    if (!mean || !std_dev || !mm) {
        std::fprintf(stderr, "malloc failed\n");
        std::exit(1);
    }

    calcmean(input, mean, num_threads);
    calc_mm_std(input, mean, mm, std_dev, num_threads);
    pearson(mm, std_dev, output, num_threads);

    free(mean);
    free(std_dev);
    free(mm);
}

void writeoutput(double *output, long long cor_size, const char *name)
{
    FILE *f = fopen(name, "wb");
    for (long long i = 0; i < cor_size; i++) {
        std::fprintf(f, "%.15g\n", output[i]);
    }
    fclose(f);
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s matrix_height matrix_width [seed] [num_threads]\n", argv[0]);
        std::exit(-1);
    }

    ROWS = atoi(argv[1]);
    if (ROWS < 1) { std::fprintf(stderr, "error: height must be at least 1\n"); std::exit(-1); }

    COLS = atoi(argv[2]);
    if (COLS < 1) { std::fprintf(stderr, "error: width must be at least 1\n"); std::exit(-1); }

    unsigned long seed = 12345;
    if (argc >= 4) { seed = (unsigned long)atol(argv[3]); }


    int num_threads = 4;
    if (argc >= 5) num_threads = atoi(argv[4]);

    std::cout << "Threads: " << num_threads
              << ", Matrix: " << ROWS << "x" << COLS << std::endl;

    //used to generate the correct filename
    char output_filename[64];
    snprintf(output_filename, sizeof(output_filename), "pccout_%d_%d.dat", ROWS, COLS);

    long long cor_size = (long long)(ROWS - 1) * ROWS / 2;

    double *matrix = (double *)malloc(sizeof(double) * ROWS * COLS);
    double *output = (double *)malloc(sizeof(double) * cor_size);
    if (!matrix || !output) { std::fprintf(stderr, "malloc failed\n"); return 1; }

    generatematrix(matrix, seed);

    auto start = std::chrono::steady_clock::now();
    pearson_par(matrix, output, num_threads);
    auto end = std::chrono::steady_clock::now();

    std::cout << "Elapsed time =  "
              << std::fixed << std::setprecision(4)
              << std::chrono::duration<double>(end - start).count()
              << " sec\n";

    writeoutput(output, cor_size, output_filename);

    free(matrix);
    free(output);
    return 0;
}