#include "parallel-cross-ppi.hpp"
#include "utils.hpp"
int main(int argc, char **argv)
{
    int rank, size;

    // Check for correct number of arguments
    if (argc != 3)
    {
        if (rank == 0)
        {
            std::cerr << "Usage: mpirun -np <num_processes> <executable> <method> <alpha>" << std::endl;
            std::cerr << "<method>: quantile or mean" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    // 记录程序开始的时间
    auto start_time = std::chrono::high_resolution_clock::now();

    // Initialize MPI
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Parse command line arguments
    std::string method = argv[1];      // "quantile" or "mean"
    double alpha = std::stod(argv[2]); // Significance level for CI

    // Example data initialization (rank 0 initializes, then broadcasts)
    std::vector<double> Y, Y_hat, Y_hat_unlabeled;
    if (rank == 0)
    {
        // Y = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0};
        // Y_hat = {1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 11.1, 12.1, 13.1, 14.1, 15.1};
        // Y_hat_unlabeled = {1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2, 9.2, 10.2, 11.2, 12.2, 13.2, 14.2, 15.2, 16.2, 17.2, 18.2, 19.2, 20.2};
        loadData(Y, Y_hat, Y_hat_unlabeled);
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

    // Compute the cross-ppi confidence interval based on selected method
    std::pair<double, double> ci;
    if (method == "quantile")
    {
        double q = 0.75; // Example quantile (75th percentile)
        ci = parallel_cross_ppi_quantile_ci(Y, Y_hat, Y_hat_unlabeled, rank, size, q, alpha);

        // Output: What kind of estimate and which quantile
        if (rank == 0)
        {
            std::cout << "Quantile-based Confidence Interval (Quantile: " << q * 100 << "%): ["
                      << ci.first << ", " << ci.second << "]" << std::endl;
        }
    }
    else if (method == "mean")
    {
        ci = parallel_cross_ppi_mean_ci(Y, Y_hat, Y_hat_unlabeled, rank, size, alpha);

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