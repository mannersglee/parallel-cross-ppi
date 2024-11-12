#include <mpi.h>
#include <vector>
#include <cmath>
#include <numeric>
#include <iostream>
#include <algorithm>
#include <limits>
#include <cassert>
#include <boost/math/distributions/normal.hpp>
#include <chrono>

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

// Parallel mean computation using MPI with different methods
double parallel_mean(const std::vector<double> &data, int rank, int size, const std::string &parallel_method)
{
    int n = data.size();
    int local_n = n / size;
    int start = rank * local_n;
    int end = (rank == size - 1) ? n : start + local_n;

    double local_sum = partial_sum(data, start, end);
    double total_sum = 0.0;

    if (parallel_method == "reduce")
    {
        MPI_Reduce(&local_sum, &total_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0)
        {
            total_sum /= n;
        }
    }
    else if (parallel_method == "scatter")
    {
        std::vector<double> recv_data(local_n);
        MPI_Scatter(data.data(), local_n, MPI_DOUBLE, recv_data.data(), local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        local_sum = std::accumulate(recv_data.begin(), recv_data.end(), 0.0);
        MPI_Reduce(&local_sum, &total_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0)
        {
            total_sum /= n;
        }
    }
    else if (parallel_method == "gather")
    {
        MPI_Gather(&local_sum, 1, MPI_DOUBLE, &total_sum, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0)
        {
            total_sum /= n;
        }
    }
    else
    {
        if (rank == 0)
        {
            std::cerr << "Invalid parallel method: " << parallel_method << ". Choose 'reduce', 'scatter', or 'gather'." << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    return total_sum;
}

// Parallel variance computation using MPI with different methods
double parallel_variance(const std::vector<double> &data, double mean, int rank, int size, const std::string &parallel_method)
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

    if (parallel_method == "reduce")
    {
        MPI_Reduce(&local_sum_sq_diff, &total_sum_sq_diff, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0)
        {
            total_sum_sq_diff /= (n - 1);
        }
    }
    else if (parallel_method == "scatter")
    {
        std::vector<double> recv_data(local_n);
        MPI_Scatter(data.data(), local_n, MPI_DOUBLE, recv_data.data(), local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        local_sum_sq_diff = 0.0;
        for (const auto &val : recv_data)
        {
            local_sum_sq_diff += (val - mean) * (val - mean);
        }
        MPI_Reduce(&local_sum_sq_diff, &total_sum_sq_diff, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0)
        {
            total_sum_sq_diff /= (n - 1);
        }
    }
    else if (parallel_method == "gather")
    {
        MPI_Gather(&local_sum_sq_diff, 1, MPI_DOUBLE, &total_sum_sq_diff, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0)
        {
            total_sum_sq_diff /= (n - 1);
        }
    }
    else
    {
        if (rank == 0)
        {
            std::cerr << "Invalid parallel method: " << parallel_method << ". Choose 'reduce', 'scatter', or 'gather'." << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    return total_sum_sq_diff;
}

// Function to calculate the cross-ppi confidence interval for the mean
std::pair<double, double> parallel_cross_ppi_mean_ci(
    const std::vector<double> &Y,
    const std::vector<double> &Y_hat,
    const std::vector<double> &Y_hat_unlabeled,
    int rank, int size, double alpha, const std::string &parallel_method)
{
    int n = Y.size();
    int N = Y_hat_unlabeled.size();

    // Compute means in parallel
    double mean_Y_hat_unlabeled = parallel_mean(Y_hat_unlabeled, rank, size, parallel_method);
    double mean_Y_hat = parallel_mean(Y_hat, rank, size, parallel_method);
    double mean_Y = parallel_mean(Y, rank, size, parallel_method);
    double mean_debias = mean_Y_hat - mean_Y;

    // Cross-prediction estimator
    double cross_prediction_estimator = mean_Y_hat_unlabeled - mean_debias;

    // Compute variances in parallel
    double variance_Y_hat_unlabeled = parallel_variance(Y_hat_unlabeled, mean_Y_hat_unlabeled, rank, size, parallel_method);
    double variance_debias = parallel_variance(Y_hat, mean_debias, rank, size, parallel_method);

    // Combine variances for confidence interval calculation
    double combined_variance = (variance_Y_hat_unlabeled / N) + (variance_debias / n);

    // Confidence interval calculation (only on rank 0)
    boost::math::normal dist(0.0, 1.0);                      // Standard normal distribution (mean=0, std=1)
    double z = boost::math::quantile(dist, 1.0 - alpha / 2); // Inverse CDF (ppf)

    double margin_of_error = z * sqrt(combined_variance);
    double lower_bound = cross_prediction_estimator - margin_of_error;
    double upper_bound = cross_prediction_estimator + margin_of_error;

    if (rank == 0)
    {
        return {lower_bound, upper_bound};
    }
    return {0.0, 0.0}; // Other ranks do not need to return a value
}

// Function to calculate the cross-ppi confidence interval for quantile estimation
std::pair<double, double> parallel_cross_ppi_quantile_ci(
    const std::vector<double> &Y,
    const std::vector<double> &Y_hat,
    const std::vector<double> &Y_hat_unlabeled,
    int rank, int size, double q, double alpha, const std::string &parallel_method)
{
    int n = Y.size();
    int N = Y_hat_unlabeled.size();

    // Compute the mean of Y (labeled data) and Y_hat (predicted labels)
    double mean_Y = parallel_mean(Y, rank, size, parallel_method);
    double mean_Y_hat = parallel_mean(Y_hat, rank, size, parallel_method);

    // Compute the variance of Y (labeled data) and Y_hat (predicted labels)
    double var_Y = parallel_variance(Y, mean_Y, rank, size, parallel_method);
    double var_Y_hat = parallel_variance(Y_hat, mean_Y_hat, rank, size, parallel_method);

    // Sort Y_hat_unlabeled (unlabeled predictions) for quantile calculation
    std::vector<double> sorted_Y_hat_unlabeled = Y_hat_unlabeled;
    std::sort(sorted_Y_hat_unlabeled.begin(), sorted_Y_hat_unlabeled.end());

    // Quantile index for the sorted array (for both labeled and unlabeled data)
    int q_idx_unlabeled = static_cast<int>(q * N);
    int q_idx_labeled = static_cast<int>(q * n);

    // Bootstrap method: get resamples for variance estimation (simplified version)
    double ci_l = std::numeric_limits<double>::infinity();
    double ci_u = -std::numeric_limits<double>::infinity();

    // Use the sorted predictions for calculating quantiles
    double lower_bound = sorted_Y_hat_unlabeled[q_idx_unlabeled];
    double upper_bound = sorted_Y_hat_unlabeled[N - 1 - q_idx_unlabeled];

    // Compute the standard deviation of the quantile estimates
    double std_dev = std::sqrt(var_Y / n + var_Y_hat / N);

    // Confidence interval calculation based on quantile and alpha level
    boost::math::normal dist(0.0, 1.0);                      // Standard normal distribution (mean=0, std=1)
    double z = boost::math::quantile(dist, 1.0 - alpha / 2); // Inverse CDF (ppf)

    ci_l = lower_bound - z * std_dev;
    ci_u = upper_bound + z * std_dev;

    return std::make_pair(ci_l, ci_u);
}
