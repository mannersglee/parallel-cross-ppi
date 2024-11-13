#include <vector>
#include <cmath>
#include <numeric>
#include <iostream>
#include <algorithm>
#include <limits>
#include <cassert>
#include <boost/math/distributions/normal.hpp>
#include <chrono>

// Function to compute the partial sum of a vector segment (串行)
double partial_sum(const std::vector<double> &data, int start, int end)
{
    double sum = 0.0;
    for (int i = start; i < end; ++i)
    {
        sum += data[i];
    }
    return sum;
}

// Serial mean computation
double serial_mean(const std::vector<double> &data)
{
    int n = data.size();
    double sum = partial_sum(data, 0, n);
    return sum / n;
}

// Serial variance computation
double serial_variance(const std::vector<double> &data, double mean)
{
    int n = data.size();
    double sum_sq_diff = 0.0;
    for (int i = 0; i < n; ++i)
    {
        sum_sq_diff += (data[i] - mean) * (data[i] - mean);
    }
    return sum_sq_diff / (n - 1);
}

// Function to calculate the cross-ppi confidence interval for the mean (串行版本)
std::pair<double, double> serial_cross_ppi_mean_ci(
    const std::vector<double> &Y,
    const std::vector<double> &Y_hat,
    const std::vector<double> &Y_hat_unlabeled,
    double alpha)
{
    int n = Y.size();
    int N = Y_hat_unlabeled.size();

    // Compute means serially
    double mean_Y_hat_unlabeled = serial_mean(Y_hat_unlabeled);
    double mean_Y_hat = serial_mean(Y_hat);
    double mean_Y = serial_mean(Y);
    double mean_debias = mean_Y_hat - mean_Y;

    // Cross-prediction estimator
    double cross_prediction_estimator = mean_Y_hat_unlabeled - mean_debias;

    // Compute variances serially
    double variance_Y_hat_unlabeled = serial_variance(Y_hat_unlabeled, mean_Y_hat_unlabeled);
    double variance_debias = serial_variance(Y_hat, mean_debias);

    // Combine variances for confidence interval calculation
    double combined_variance = (variance_Y_hat_unlabeled / N) + (variance_debias / n);

    // Confidence interval calculation
    boost::math::normal dist(0.0, 1.0);                      // Standard normal distribution (mean=0, std=1)
    double z = boost::math::quantile(dist, 1.0 - alpha / 2); // Inverse CDF (ppf)

    double margin_of_error = z * sqrt(combined_variance);
    double lower_bound = cross_prediction_estimator - margin_of_error;
    double upper_bound = cross_prediction_estimator + margin_of_error;

    return {lower_bound, upper_bound};
}
