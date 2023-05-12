#!/bin/bash

# Array of arguments
args=(48 36 27 18 12 8 4 2 1)

# Loop through the arguments and submit SLURM job
for arg in "${args[@]}"; do
		echo "Running with $arg processes:"
    sbatch -n "$arg" run_benchmarks.slurm
done

