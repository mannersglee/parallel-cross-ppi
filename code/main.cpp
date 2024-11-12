#include "parallel-cross-ppi.hpp"
#include "utils.hpp"
#include <cstring> // for strcmp

int main(int argc, char **argv)
{
    int rank, size;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Check for correct number of arguments
    if (argc != 5)
    {
        if (rank == 0)
        {
            std::cerr << "Usage: mpirun -np <num_processes> <executable> <method> <alpha> <parallel_method>" << std::endl;
            std::cerr << "<method>: quantile or mean" << std::endl;
            std::cerr << "<parallel_method>: reduce, scatter, gather" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    // Parse command line arguments
    std::string method = argv[1];      // "quantile" or "mean"
    double alpha = std::stod(argv[2]); // Significance level for CI
    int repeat = std::stoi(argv[3]);   // Number of repeats for data loading
    std::string parallel_method = argv[4]; // "reduce", "scatter", or "gather"

    // Example data initialization (rank 0 initializes, then broadcasts)
    std::vector<double> Y, Y_hat, Y_hat_unlabeled;
    if (rank == 0)
    {
        loadData(Y, Y_hat, Y_hat_unlabeled, repeat);
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

    // 记录程序开始的时间
    auto start_time = std::chrono::high_resolution_clock::now();

    // Compute the cross-ppi confidence interval based on selected method
    std::pair<double, double> ci;
    if (method == "quantile")
    {
        double q = 0.75; // Example quantile (75th percentile)
        ci = parallel_cross_ppi_quantile_ci(Y, Y_hat, Y_hat_unlabeled, rank, size, q, alpha, parallel_method);

        // Output: What kind of estimate and which quantile
        if (rank == 0)
        {
            std::cout << "Quantile-based Confidence Interval (Quantile: " << q * 100 << "%): ["
                      << ci.first << ", " << ci.second << "]" << std::endl;
        }
    }
    else if (method == "mean")
    {
        ci = parallel_cross_ppi_mean_ci(Y, Y_hat, Y_hat_unlabeled, rank, size, alpha, parallel_method);

        // Output: Mean-based estimate
        if (rank == 0)
        {
            std::cout << "Mean-based Confidence Interval: ["
                      << ci.first << ", " << ci.second << "]" << std::endl;
        }
    }
    else
    {
        if (rank == 0)
        {
            std::cerr << "Invalid method: " << method << ". Choose 'quantile' or 'mean'." << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    // Finalize MPI
    MPI_Finalize();

    // 记录程序结束的时间
    auto end_time = std::chrono::high_resolution_clock::now();

    // 计算并输出运行时间
    std::chrono::duration<double> duration = end_time - start_time;
    if (rank == 0)
    {
        std::cout << "Program execution time (size " << size << "): " << duration.count() << " seconds." << std::endl;
    }

    return 0;
}
