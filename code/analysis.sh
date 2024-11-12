#!/bin/bash

# Set the alpha value
alpha=0.05

# Set repeat nums
repeat=10

# parallel_method: "reduce", "scatter", or "gather"
parallel_method="gather"

# Compile the program
make main

# Loop over different MPI sizes (1 to 8)
for size in {1..8}; do
    echo "Running with np=$size, alpha=$alpha, repeat=$repeat, parallel_method=$parallel_method"

    # Run quantile estimation
    mpirun -np $size ./main quantile $alpha $repeat $parallel_method

    # Run mean estimation
    mpirun -np $size ./main mean $alpha $repeat $parallel_method
done

# Clean up
make clean
