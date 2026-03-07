#include <vector>
#include <algorithm>
#include <iostream>
#include <chrono>
// std::thread for spawning worker threads
#include <thread>
// std::barrier for synchronizing threads between phases
#include <barrier>

// Parallelised odd-even sort using std::thread and std::barrier 
// Each thread handles a chunk of the comparisons, barrier syncs between phases
void oddeven_sort_par(std::vector<int>& numbers, int num_threads)
{
    int s = static_cast<int>(numbers.size());
    
    // Create barrier with num_threads participants - all must arrive before any can proceed
    std::barrier sync_point(num_threads);
    
    // Lambda function executed by each worker thread
    auto worker = [&](int thread_id) {
        for (int phase = 1; phase <= s; phase++) {
            int start_idx = phase % 2;
            
            // Calculate which pairs this thread is responsible for
            int total_pairs = (s - 1 - start_idx + 2) / 2;
            // Divide pairs evenly among threads (static distribution)
            int pairs_per_thread = (total_pairs + num_threads - 1) / num_threads;
            // Each thread gets a unique range based on thread_id
            int my_start = thread_id * pairs_per_thread;
            int my_end = std::min(my_start + pairs_per_thread, total_pairs);
            
            // Process assigned pairs (same logic as sequential, but only for this thread's portion)
            for (int p = my_start; p < my_end; p++) {
                int j = start_idx + p * 2;
                if (j < s - 1 && numbers[j] > numbers[j + 1]) {
                    std::swap(numbers[j], numbers[j + 1]);
                }
            }
            
            // Barrier synchronization - wait for all threads before next phase
            sync_point.arrive_and_wait();
        }
    };
    
    // Create thread pool and launch worker threads
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    // Spawn num_threads threads, each running the worker lambda
    for (int t = 0; t < num_threads; t++) {
        threads.emplace_back(worker, t);
    }
    
    // Wait for all threads to complete before returning
    for (auto& th : threads) {
        th.join();
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
    int size = 100000;
    int num_threads = 4;
    
    if (argc > 1) num_threads = std::stoi(argv[1]);
    if (argc > 2) size = std::stoi(argv[2]);
    
    std::vector<int> numbers(size);
    srand(time(0));
    std::generate(numbers.begin(), numbers.end(), rand);

    print_sort_status(numbers);
    
    std::cout << "Array size: " << size << ", Threads: " << num_threads << "\n";
    
    auto start = std::chrono::steady_clock::now();
    oddeven_sort_par(numbers, num_threads);
    auto end = std::chrono::steady_clock::now();
    
    print_sort_status(numbers);
    std::cout << "Elapsed time = " << std::chrono::duration<double>(end - start).count() << " sec\n";
    
    return 0;
}