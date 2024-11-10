#!/bin/bash

# Set the alpha value
alpha=0.05

# Compile the program
make main

# Loop over different MPI sizes (1 to 8)
for size in {1..8}; do
    echo "Running with np=$size"

    # Run quantile estimation
    mpirun -np $size ./main quantile $alpha

    # Run mean estimation
    mpirun -np $size ./main mean $alpha
done

# Clean up
make clean
