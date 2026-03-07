# Sequential versions are provided. Students parallelize and add *_par sources.

CC	= gcc
CXX	= g++

# macOS: Use -Xpreprocessor for OpenMP with Apple clang + Homebrew libomp
# Linux: Change to just -fopenmp
OPENMP_FLAGS = -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp

all: oddevensort pcc_seq verify

# --- Sequential (given) ---
oddevensort: oddevensort.cpp
	$(CXX) oddevensort.cpp -std=c++17 -o oddevensort

pcc_seq: pcc_seq.cpp
	$(CXX) pcc_seq.cpp -lm -std=c++17 -o pcc_seq

# --- Parallel OpenMP versions ---
oddevensort_par: oddevensort_par.cpp
	$(CXX) oddevensort_par.cpp $(OPENMP_FLAGS) -std=c++17 -o oddevensort_par

pcc_par: pcc_seq_par.cpp
	$(CXX) pcc_seq_par.cpp $(OPENMP_FLAGS) -lm -std=c++17 -o pcc_par

# --- Parallel std::thread versions ---
oddevensort_std: oddevensort_std.cpp
	$(CXX) oddevensort_std.cpp -std=c++20 -o oddevensort_std

pcc_std: pcc_seq_std.cpp
	$(CXX) pcc_seq_std.cpp -lm -std=c++20 -o pcc_std

# PCC output verifier (floating-point comparison)
verify: verify.c
	$(CC) verify.c -lm -std=c99 -o verify -DQUIET

# Run benchmark: validate correctness and report speedup (seq vs par)
benchmark-pcc: pcc_seq pcc_par verify
	./benchmark.sh ./pcc_seq ./pcc_par ./verify

clean:
	rm -rf oddevensort oddevensort_par oddevensort_std pcc_seq pcc_par pcc_std verify *.dat
