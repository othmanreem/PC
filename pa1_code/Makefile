# Sequential versions are provided. Students parallelize and add *_par sources.

CC	= gcc
CXX	= g++

all: oddevensort pcc_seq verify

# --- Sequential (given) ---
oddevensort: oddevensort.cpp
	$(CXX) oddevensort.cpp -std=c++17 -o oddevensort

pcc_seq: pcc_seq.cpp
	$(CXX) pcc_seq.cpp -lm -std=c++17 -o pcc_seq

# --- Parallel (students add oddevensort_par.cpp and pcc_par.c) ---
oddevensort_par: oddevensort_par.cpp
	$(CXX) oddevensort_par.cpp -std=c++17 -o oddevensort_par

pcc_par: pcc_par.c
	$(CC) pcc_par.c -lm -std=c99 -o pcc_par

# PCC output verifier (floating-point comparison)
verify: verify.c
	$(CC) verify.c -lm -std=c99 -o verify -DQUIET

# Run benchmark: validate correctness and report speedup (seq vs par)
benchmark-pcc: pcc_seq pcc_par verify
	./benchmark.sh ./pcc_seq ./pcc_par ./verify

clean:
	rm -rf oddevensort oddevensort_par pcc_seq pcc_par verify *.dat
