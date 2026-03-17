#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <iostream>
#include <iomanip>

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
 * Calculate row mean
 */
void calcmean(double *matrix, double *mean){
	int i,j;
	double sum;

	for(i = 0; i < ROWS; i++){
		sum = 0.0;
		for(j = 0; j < COLS; j++){
			sum += matrix[i * COLS + j];
		}
		mean[i] = sum / (double)COLS;
	}
}

/**
 * Calculate matrix - rowmean, and standard deviation for every row 
 */
void calc_mm_std(double *matrix, double *mean, double *mm, double *std){
	int i,j;
	double sum, diff;

	for(i = 0; i < ROWS; i++){
		sum = 0.0;
		for(j = 0; j < COLS; j++){
			diff = matrix[i * COLS + j] - mean[i];
			mm[i * COLS + j] = diff;
			sum += diff * diff;
		}
		std[i] = std::sqrt(sum);
	}
}

void pearson(double *mm, double *std, double *output){
	int i, sample1, sample2;
	double sum, r;

	for(sample1 = 0; sample1 < ROWS-1; sample1++){
		int summ = 0;
		for(int l = 0; l <= sample1+1; l++)
			summ += l;

		for(sample2 = sample1+1; sample2 < ROWS; sample2++){
			sum = 0.0;
			for(i = 0; i < COLS; i++){
				sum += mm[sample1 * COLS + i] * mm[sample2 * COLS + i];
			}
			r = sum / (std[sample1] * std[sample2]);
			output[sample1 * ROWS + sample2 - summ] = r;
		}
	}
}

void pearson_seq(double *input, double *output, int cor_size){
    
    double *mean = (double*)malloc(sizeof(double) * ROWS);
	double *std  = (double*)malloc(sizeof(double) * ROWS);
	
	if(mean == NULL || std == NULL){
        std::fprintf(stderr, "did exit\n");
		std::exit(0);
	}
    double *minusmean = (double*)malloc(sizeof(double) * ROWS * COLS);
	if(minusmean == NULL) {
        std::fprintf(stderr, "did exit\n");
		std::exit(0);
	}
    
    calcmean(input, mean);
	calc_mm_std(input, mean, minusmean, std);
	pearson(minusmean, std, output);

    free(mean);
    free(minusmean);
    free(std);
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
	
	if (argc < 3) { std::fprintf(stderr, "usage: %s matrix_height matrix_width [seed]\n", argv[0]); std::exit(-1); }

	ROWS = atoi(argv[1]);
	if (ROWS < 1) { std::fprintf(stderr, "error: height must be at least 1\n"); std::exit(-1); }

	COLS = atoi(argv[2]);
	if (COLS < 1) { std::fprintf(stderr, "error: width must be at least 1\n"); std::exit(-1); }

	unsigned long seed = 12345;
	if (argc >= 4) { seed = (unsigned long)atol(argv[3]); }

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
	pearson_seq(matrix, output, cor_size);
	auto end = std::chrono::steady_clock::now();
	std::cout << "Elapsed time =  " << std::fixed << std::setprecision(4) << std::chrono::duration<double>(end - start).count() << " sec\n";

	writeoutput(output, cor_size, output_filename);	

	free(output);
	free(matrix);
	return(0);
}
