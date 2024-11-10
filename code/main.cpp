#include <mpi.h>
#include <vector>
#include <cmath>
#include <numeric>
#include <iostream>

#include "statistic-test.hpp"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Example data initialization (rank 0 initializes, then broadcasts)
    std::vector<double> Y, Y_hat, Y_hat_unlabeled;
    if (rank == 0)
    {
        Y = {1.0, 2.0, 3.0, 4.0, 5.0};
        Y_hat = {1.1, 2.1, 3.1, 4.1, 5.1};
        Y_hat_unlabeled = {1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2, 9.2, 10.2};
    }

    // Broadcast the sizes to all processes
    int n = 0, N = 0;
    if (rank == 0)
    {
        n = Y.size();
        N = Y_hat_unlabeled.size();
    }
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Resize and broadcast the data to all processes
    if (rank != 0)
    {
        Y.resize(n);
        Y_hat.resize(n);
        Y_hat_unlabeled.resize(N);
    }
    MPI_Bcast(Y.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(Y_hat.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(Y_hat_unlabeled.data(), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Compute the cross-ppi confidence interval
    auto ci = parallel_cross_ppi_mean_ci(Y, Y_hat, Y_hat_unlabeled, rank, size);

    // Print results on rank 0
    if (rank == 0)
    {
        std::cout << "Confidence Interval: [" << ci.first << ", " << ci.second << "]" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
