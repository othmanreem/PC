#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <iostream>
#include <iomanip>
// std::thread for spawning worker threads
#include <thread>
#include <vector>
// std::atomic for lock free dynamic work distribution
#include <atomic>

int COLS = 128;
int ROWS = 128;

/**
 * Generate matrix in-place using seeded drand48() for reproducible data.
 **/
void generatematrix(double *matrix, unsigned long seed){
	srand48((long)seed);
	for (int i = 0; i < ROWS * COLS; i++) {
		matrix[i] = drand48();
	}
}

/**
 * Calculate row mean - parallelised over rows using std::thread
 */
void calcmean(double *matrix, double *mean, int num_threads){
	// each thread processes a range of rows
	auto worker = [&](int start_row, int end_row) {
		for (int i = start_row; i < end_row; i++) {
			double sum = 0.0;
			for (int j = 0; j < COLS; j++) {
				sum += matrix[i * COLS + j];
			}
			mean[i] = sum / (double)COLS;
		}
	};
	
	// Thread storage vector
	std::vector<std::thread> threads;
	// Static distribution: divide rows evenly among threads
	int rows_per_thread = (ROWS + num_threads - 1) / num_threads;
	
	// Spawn threads, each handling a chunk of rows
	for (int t = 0; t < num_threads; t++) {
		int start = t * rows_per_thread;
		int end = std::min(start + rows_per_thread, ROWS);
		if (start < end) {
			// Launch thread with its assigned row range
			threads.emplace_back(worker, start, end);
		}
	}
	// Wait for all threads to complete
	for (auto& th : threads) th.join();
}

/**
 * Calculate matrix - rowmean, and standard deviation for every row 
 */
void calc_mm_std(double *matrix, double *mean, double *mm, double *std_dev, int num_threads){
	// each thread processes a range of rows
	auto worker = [&](int start_row, int end_row) {
		for (int i = start_row; i < end_row; i++) {
			double sum = 0.0;
			for (int j = 0; j < COLS; j++) {
				double diff = matrix[i * COLS + j] - mean[i];
				mm[i * COLS + j] = diff;
				sum += diff * diff;
			}
			std_dev[i] = std::sqrt(sum);
		}
	};
	
	// Thread storage vector
	std::vector<std::thread> threads;
	// Static distribution: divide rows evenly among threads
	int rows_per_thread = (ROWS + num_threads - 1) / num_threads;
	
	// Spawn threads, each handling a chunk of rows
	for (int t = 0; t < num_threads; t++) {
		int start = t * rows_per_thread;
		int end = std::min(start + rows_per_thread, ROWS);
		if (start < end) {
			// Launch thread with its assigned row range
			threads.emplace_back(worker, start, end);
		}
	}
	// Wait for all threads to complete
	for (auto& th : threads) th.join();
}

/**
 * Pearson correlation - parallelised with dynamic work stealing via atomic counter
 */
void pearson(double *mm, double *std_dev, double *output, int num_threads){
	// Precompute output offsets for each sample1
	long long *offsets = (long long *)malloc(sizeof(long long) * ROWS);
	{
		long long acc = 0;
		for (int s1 = 0; s1 < ROWS - 1; s1++) {
			offsets[s1] = acc;
			acc += (ROWS - 1 - s1);
		}
	}
	
	// Atomic counter for dynamic work distribution (like OpenMP schedule(dynamic))
	std::atomic<int> next_row(0);
	
	// Worker lambda - threads grab rows dynamically via atomic fetch_add
	auto worker = [&]() {
		while (true) {
			// Atomically claim the next row to process (lock-free work stealing)
			int sample1 = next_row.fetch_add(1);
			// Exit when all rows have been claimed
			if (sample1 >= ROWS - 1) break;
			
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
	};
	
	// Thread storage vector
	std::vector<std::thread> threads;
	// Spawn worker threads
	for (int t = 0; t < num_threads; t++) {
		threads.emplace_back(worker);
	}
	// Wait for all threads to complete
	for (auto& th : threads) th.join();
	
	free(offsets);
}

void pearson_par(double *input, double *output, int num_threads){
    
    double *mean = (double*)malloc(sizeof(double) * ROWS);
	double *std_dev = (double*)malloc(sizeof(double) * ROWS);
	
	if(mean == NULL || std_dev == NULL){
        std::fprintf(stderr, "did exit\n");
		std::exit(0);
	}
    double *minusmean = (double*)malloc(sizeof(double) * ROWS * COLS);
	if(minusmean == NULL) {
        std::fprintf(stderr, "did exit\n");
		std::exit(0);
	}
    
    calcmean(input, mean, num_threads);
	calc_mm_std(input, mean, minusmean, std_dev, num_threads);
	pearson(minusmean, std_dev, output, num_threads);

    free(mean);
    free(minusmean);
    free(std_dev);
}

void writeoutput(double *output, int cor_size, char *name)
{
	FILE *f;

	f = fopen(name,"wb");
	for (int i = 0; i < cor_size; i++) {
		std::fprintf(f, "%.15g\n", output[i]);
	}
	fclose(f);
}

int main(int argc, char **argv){
	
	if (argc < 3) { std::fprintf(stderr, "usage: %s matrix_height matrix_width [seed] [num_threads]\n", argv[0]); std::exit(-1); }

	ROWS = atoi(argv[1]);
	if (ROWS < 1) { std::fprintf(stderr, "error: height must be at least 1\n"); std::exit(-1); }

	COLS = atoi(argv[2]);
	if (COLS < 1) { std::fprintf(stderr, "error: width must be at least 1\n"); std::exit(-1); }

	unsigned long seed = 12345;
	if (argc >= 4) { seed = (unsigned long)atol(argv[3]); }

	int num_threads = 4;
	if (argc >= 5) { num_threads = atoi(argv[4]); }

	std::cout << "Threads: " << num_threads << ", Matrix: " << ROWS << "x" << COLS << std::endl;

	//used to generate the correct filename
	char output_filename[30];
	snprintf(output_filename, 30, "pccout_%d_%d.dat", ROWS, COLS);
	
	//calculates the size of the output
	long long cor_size = ROWS - 1;
    cor_size *= ROWS;
    cor_size /= 2;

	double *matrix, *output;
	output = (double*)malloc(sizeof(double) * cor_size);
	matrix = (double*)malloc(sizeof(double) * COLS * ROWS);

	if(matrix == NULL){
		return(1);
	}
	
	generatematrix(matrix, seed);

	/* Chrono timer (same style as oddevensort) */
	auto start = std::chrono::steady_clock::now();
	pearson_par(matrix, output, num_threads);
	auto end = std::chrono::steady_clock::now();
	std::cout << "Elapsed time =  " << std::fixed << std::setprecision(4) << std::chrono::duration<double>(end - start).count() << " sec\n";

	writeoutput(output, cor_size, output_filename);	

	free(output);
	free(matrix);
	return(0);
}
