#include <vector>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <thread>
#include <barrier>

// Parallelized odd-even sort algorithm using C++ threads
// The algorithm alternates between odd and even phases where independent pairs
// of elements can be compared and swapped in parallel within each phase.
void oddeven_sort_par(std::vector<int>& numbers, unsigned int num_threads = 4)
{
    auto s = numbers.size();
    
    // Create a barrier to synchronize threads between phases
    std::barrier<> phase_barrier(num_threads);
    
    // Function that each thread executes
    auto thread_work = [&](unsigned int thread_id) {
        // Iterate through all phases
        for (int i = 1; i <= s; i++) {
            // Each thread handles a subset of the comparisons
            // Start position for this phase
            int start = i % 2;
            
            // Each thread processes every (2 * num_threads)-th pair
            // This distributes the work evenly across threads
            for (int j = start + thread_id * 2; j < s - 1; j = j + 2 * num_threads) {
                if (numbers[j] > numbers[j + 1]) {
                    std::swap(numbers[j], numbers[j + 1]);
                }
            }
            
            // Wait for all threads to finish this phase before moving to the next
            phase_barrier.arrive_and_wait();
        }
    };
    
    // Create and launch all threads
    std::vector<std::thread> threads;
    for (unsigned int i = 0; i < num_threads; i++) {
        threads.emplace_back(thread_work, i);
    }
    
    // Wait for all threads to complete
    for (auto& t : threads) {
        t.join();
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
    constexpr unsigned int size = 100000; // Number of elements in the input
    
    // Determine number of threads from command line or default to 4
    unsigned int num_threads = 4;
    if (argc > 1) {
        num_threads = std::stoi(argv[1]);
    }
    
    // Initialize a vector with integers of value 0
    std::vector<int> numbers(size);
    
    // Populate our vector with (pseudo)random numbers
    srand(time(0));
    std::generate(numbers.begin(), numbers.end(), rand);
    
    print_sort_status(numbers);
    
    std::cout << "Running with " << num_threads << " threads\n";
    
    auto start = std::chrono::steady_clock::now();
    oddeven_sort_par(numbers, num_threads);
    auto end = std::chrono::steady_clock::now();
    
    print_sort_status(numbers);
    
    std::cout << "Elapsed time = " 
              << std::chrono::duration<double>(end - start).count() 
              << " sec\n";
    
    return 0;
}