#!/bin/bash

# CUDA Benchmark Script for PA2
# Tests both odd-even sort and PCC CUDA implementations

GREEN="\033[0;32m"
RED="\033[0;31m"
YELLOW="\033[0;33m"
CYAN="\033[0;36m"
END="\033[0m"

echo "=============================================="
echo -e "${CYAN}CUDA Benchmark: Odd-Even Sort & PCC${END}"
echo "=============================================="
echo ""

# Check if CUDA executables exist
if [ ! -f "./oddevensort_cuda" ]; then
    echo -e "${RED}Error: oddevensort_cuda not found. Run 'make cuda' first.${END}"
    exit 1
fi

if [ ! -f "./pcc_cuda" ]; then
    echo -e "${RED}Error: pcc_cuda not found. Run 'make cuda' first.${END}"
    exit 1
fi

# ============================================
# Part 1: Odd-Even Sort Benchmark
# ============================================

echo -e "${CYAN}=== Part 1: Odd-Even Sort ===${END}"
echo ""

# ── Sequential reference ──────────────────────────────────────────
echo -e "${CYAN}=== SEQUENTIAL ODD-EVEN SORT ===${END}"
if [ ! -f "./oddevensort" ]; then
    echo -e "${RED}Error: oddevensort not found. Run 'make' first.${END}"
else
    seq_ref_output=$(./oddevensort 2>&1)
    echo "$seq_ref_output"
    seq_time=$(echo "$seq_ref_output" \
        | grep "Elapsed time" \
        | sed 's/.*Elapsed time =  *\([0-9.]*\).*/\1/')
fi

echo ""

# ── CUDA variants ─────────────────────────────────────────────────
echo -e "${CYAN}=== CUDA ODD-EVEN SORT ===${END}"
echo "Running oddevensort_cuda (single-block and multi-block variants)..."
echo ""

cuda_sort_output=$(./oddevensort_cuda 2>&1)
echo "$cuda_sort_output"

# ── Speedup ───────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}=== SPEEDUP SUMMARY ===${END}"

multi_time_100k=$(echo "$cuda_sort_output" \
    | grep "\[Multi-block\].*n=100000" \
    | sed 's/.*Time=\([0-9.]*\)s.*/\1/')

if [ -n "$seq_time" ] && [ -n "$multi_time_100k" ]; then
    speedup=$(awk -v s="$seq_time" -v c="$multi_time_100k" \
        'BEGIN { printf "%.2f", s/c }')
    echo "  Sequential (100K):         ${seq_time}s"
    echo "  CUDA multi-block (100K):   ${multi_time_100k}s"
    echo -e "  ${YELLOW}Speedup: ${speedup}x${END}"
else
    echo "  Sequential (100K): ${seq_time}s"
    echo "  (Multi-block 100K time not found — check CUDA binary output format)"
fi

echo ""

# ============================================
# Part 2: PCC Benchmark (CUDA vs Sequential)
# ============================================

echo -e "${CYAN}=== Part 2: Pearson Correlation Coefficient ===${END}"
echo ""

SEED=42

# Build sequential and verify if needed
if [ ! -f "./pcc_seq" ]; then
    echo "Building pcc_seq..."
    make pcc_seq
fi

if [ ! -f "./verify" ]; then
    echo "Building verify..."
    make verify
fi

echo "PCC benchmark: validation + speedup (seq vs CUDA)"
echo "-------------------------------------------------"

for n in 64 128 256 512 1024 2048 4096; do
    echo ""
    echo -e "${YELLOW}Matrix size: ${n}x${n}${END}"

    # --- Sequential run ---
    seq_output=$(./pcc_seq $n $n $SEED 2>/dev/null)
    seq_time=$(echo "$seq_output" | grep "Elapsed time" | sed -n 's/.*Elapsed time =  *\([0-9.]*\).*/\1/p')

    if [ ! -f "pccout_${n}_${n}.dat" ]; then
        echo -e "  ${RED}Sequential: Failed (no output)${END}"
        continue
    fi
    mv "pccout_${n}_${n}.dat" pccout_seq.dat

    # --- CUDA run ---
    cuda_output=$(./pcc_cuda $n $n $SEED 2>/dev/null)
    cuda_time=$(echo "$cuda_output" | grep "Elapsed time" | sed -n 's/.*Elapsed time =  *\([0-9.]*\).*/\1/p')

    if [ ! -f "pccout_${n}_${n}.dat" ]; then
        echo -e "  ${RED}CUDA: Failed (no output)${END}"
        continue
    fi
    mv "pccout_${n}_${n}.dat" pccout_cuda.dat

    # --- Validation ---
    if [ -x "./verify" ]; then
        ./verify pccout_seq.dat pccout_cuda.dat 2>/dev/null
        verify_ret=$?
        if [ "$verify_ret" -eq 0 ]; then
            echo -e "  Validation: ${GREEN}Passed (exact match)${END}"
        elif [ "$verify_ret" -eq 1 ]; then
            echo -e "  Validation: ${GREEN}Passed (within tolerance)${END}"
        else
            echo -e "  Validation: ${RED}Failed (output differs)${END}"
        fi
    else
        if diff -q pccout_seq.dat pccout_cuda.dat > /dev/null 2>&1; then
            echo -e "  Validation: ${GREEN}Passed${END}"
        else
            echo -e "  Validation: ${RED}Failed${END}"
        fi
    fi

    # --- Speedup ---
    speedup=$(awk -v s="$seq_time" -v c="$cuda_time" 'BEGIN {if (c>0 && s!="") printf "%.2f", s/c; else print "N/A"}')
    echo "  Sequential time: ${seq_time}s"
    echo "  CUDA time:       ${cuda_time}s"
    echo -e "  ${YELLOW}Speedup: ${speedup}x${END}"

    # Clean up temp files
    rm -f pccout_seq.dat pccout_cuda.dat
done

echo ""
echo "=============================================="
echo -e "${GREEN}Benchmark complete.${END}"
echo "=============================================="