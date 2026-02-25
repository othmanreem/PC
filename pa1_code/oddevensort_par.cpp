#include <vector>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <omp.h>

// Parallelised oddeven sort algo using openmp
// openmp has better implicit barriers at the end of parallel for loops
void oddeven_sort_par(std::vector<int>& numbers, int num_threads = 4)
{
    int s = static_cast<int>(numbers.size());
    
    omp_set_num_threads(num_threads);
    
    // Use a parallel region that persists, this avoids thread creation and destruction overhead for each phase
    #pragma omp parallel
    {
        for (int i = 1; i <= s; i++) {
            // phased
            int start = i % 2;
            
            // parallelise the for loop
            #pragma omp for schedule(static)
            for (int j = start; j < s - 1; j += 2) {
                if (numbers[j] > numbers[j + 1]) {
                    std::swap(numbers[j], numbers[j + 1]);
                }
            }
            // implicit barrier at end of omp for loop
        }
    }
}

void print_sort_status(const std::vector<int>& numbers)
{
    std::cout << "The input is sorted?: " 
              << (std::is_sorted(numbers.begin(), numbers.end()) ? "True" : "False") 
              << std::endl;
}

int main(int argc, char* argv[])
{
    int size = 524288;
    
    // threads num from cmd line or then 4
    int num_threads = 4;
    if (argc > 1) {
        num_threads = std::stoi(argv[1]);
    }
    if (argc > 2) {
        size = std::stoi(argv[2]);
    }
    
    // vector to hold the numbers and add randoms
    std::vector<int> numbers(size);
    srand(time(0));
    std::generate(numbers.begin(), numbers.end(), rand);
    
    print_sort_status(numbers);
    
    std::cout << "Array size: " << size << ", Threads: " << num_threads << "\n";
    
    auto start = std::chrono::steady_clock::now();
    oddeven_sort_par(numbers, num_threads);
    auto end = std::chrono::steady_clock::now();
    
    print_sort_status(numbers);
    
    std::cout << "Elapsed time = " 
              << std::chrono::duration<double>(end - start).count() 
              << " sec\n";
    
    return 0;
}