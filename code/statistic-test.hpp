#include <mpi.h>
#include <vector>
#include <cmath>
#include <numeric>
#include <iostream>

// Function to compute the partial sum of a vector segment
double partial_sum(const std::vector<double> &data, int start, int end)
{
    double sum = 0.0;
    for (int i = start; i < end; ++i)
    {
        sum += data[i];
    }
    return sum;
}

// Parallel mean computation using MPI
double parallel_mean(const std::vector<double> &data, int rank, int size)
{
    int n = data.size();
    int local_n = n / size;
    int start = rank * local_n;
    int end = (rank == size - 1) ? n : start + local_n;

    double local_sum = partial_sum(data, start, end);
    double total_sum = 0.0;
    MPI_Allreduce(&local_sum, &total_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    return total_sum / n;
}

// Parallel variance computation using MPI
double parallel_variance(const std::vector<double> &data, double mean, int rank, int size)
{
    int n = data.size();
    int local_n = n / size;
    int start = rank * local_n;
    int end = (rank == size - 1) ? n : start + local_n;

    double local_sum_sq_diff = 0.0;
    for (int i = start; i < end; ++i)
    {
        local_sum_sq_diff += (data[i] - mean) * (data[i] - mean);
    }

    double total_sum_sq_diff = 0.0;
    MPI_Allreduce(&local_sum_sq_diff, &total_sum_sq_diff, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    return total_sum_sq_diff / (n - 1);
}

// Function to calculate the cross-ppi confidence interval for the mean
std::pair<double, double> parallel_cross_ppi_mean_ci(const std::vector<double> &Y,
                                                     const std::vector<double> &Y_hat,
                                                     const std::vector<double> &Y_hat_unlabeled,
                                                     int rank, int size)
{
    int n = Y.size();
    int N = Y_hat_unlabeled.size();

    // Compute means in parallel
    double mean_Y_hat_unlabeled = parallel_mean(Y_hat_unlabeled, rank, size);
    double mean_Y_hat = parallel_mean(Y_hat, rank, size);
    double mean_Y = parallel_mean(Y, rank, size);
    double mean_debias = mean_Y_hat - mean_Y;

    // Cross-prediction estimator
    double cross_prediction_estimator = mean_Y_hat_unlabeled - mean_debias;

    // Compute variances in parallel
    double variance_Y_hat_unlabeled = parallel_variance(Y_hat_unlabeled, mean_Y_hat_unlabeled, rank, size);
    double variance_debias = parallel_variance(Y_hat, mean_debias, rank, size);

    // Combine variances for confidence interval calculation
    double combined_variance = (variance_Y_hat_unlabeled / N) + (variance_debias / n);

    // Confidence interval calculation (only on rank 0)
    double z = 1.96; // z-value for 95% confidence interval
    double margin_of_error = z * sqrt(combined_variance);
    double lower_bound = cross_prediction_estimator - margin_of_error;
    double upper_bound = cross_prediction_estimator + margin_of_error;

    if (rank == 0)
    {
        return {lower_bound, upper_bound};
    }
    return {0.0, 0.0}; // Other ranks do not need to return a value
}