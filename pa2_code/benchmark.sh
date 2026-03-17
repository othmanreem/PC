#!/bin/bash

# PCC benchmark: validate correctness (seq vs parallel output) and report speedup.

GREEN="\033[0;32m"
RED="\033[0;31m"
YELLOW="\033[0;33m"
END="\033[0m"

SEED=42

echo "PCC benchmark: validation + speedup (seq vs parallel)"
echo "-----------------------------------------------------"

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 <sequential_executable> <parallel_executable> [verify_executable]"
    echo "Example: $0 ./pcc_seq ./pcc_par ./verify"
    exit 1
fi

seq_executable=$1
par_executable=$2
verify_executable=${3:-./verify}

for n in 64 128 256 512 1024 2048 4096; do
    echo ""
    echo "n=$n (matrix ${n}x${n})"

    # --- Sequential run: writes pccout_${n}_${n}.dat ---
    seq_output=$($seq_executable $n $n $SEED 2>/dev/null)
    seq_time=$(echo "$seq_output" | grep "Elapsed time" | sed -n 's/.*Elapsed time =  *\([0-9.]*\).*/\1/p')

    if [ ! -f "pccout_${n}_${n}.dat" ]; then
        echo -e "  Validation: ${RED}Failed (seq produced no output)${END}"
        continue
    fi
    # Save seq output as pccout_seq.dat so par can write to pccout_${n}_${n}.dat (we compare seq vs par)
    mv "pccout_${n}_${n}.dat" pccout_seq.dat

    # --- Parallel run: writes pccout_${n}_${n}.dat, then we store as pccout_par.dat ---
    par_output=$($par_executable $n $n $SEED 2>/dev/null)
    par_time=$(echo "$par_output" | grep "Elapsed time" | sed -n 's/.*Elapsed time =  *\([0-9.]*\).*/\1/p')

    if [ ! -f "pccout_${n}_${n}.dat" ]; then
        echo -e "  Validation: ${RED}Failed (parallel produced no output)${END}"
    else
        mv "pccout_${n}_${n}.dat" pccout_par.dat
        # --- Validation: compare pccout_seq.dat vs pccout_par.dat ---
        if [ -x "$verify_executable" ]; then
            $verify_executable pccout_seq.dat pccout_par.dat 2>/dev/null
            verify_ret=$?
            if [ "$verify_ret" -eq 0 ]; then
                echo -e "  Validation: ${GREEN}Passed${END}"
            elif [ "$verify_ret" -eq 1 ]; then
                echo -e "  Validation: ${YELLOW}Passed (within tolerance)${END}"
            else
                echo -e "  Validation: ${RED}Failed (output differs)${END}"
            fi
        else
            if diff -q pccout_seq.dat pccout_par.dat > /dev/null 2>&1; then
                echo -e "  Validation: ${GREEN}Passed${END}"
            else
                echo -e "  Validation: ${RED}Failed (output differs)${END}"
            fi
        fi
    fi

    # --- Speedup (from program-printed elapsed times, like oddevensort) ---
    speedup=$(awk -v s="$seq_time" -v p="$par_time" 'BEGIN {if (p>0 && s!="") printf "%.2f", s/p; else print "N/A"}')
    echo -e "  Seq time: ${seq_time}s  \n  Par time: ${par_time}s  ${YELLOW} \n  Speedup: ${speedup}x${END}"
done

echo ""
echo "Done."
