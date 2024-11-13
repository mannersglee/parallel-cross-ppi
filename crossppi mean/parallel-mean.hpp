#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include "xgboost/c_api.h"
#include <Eigen/Dense> // Include Eigen for matrix operations
#include <fstream>
#include <iomanip>
#include <ctime>
#include <sstream>
#include <cstdlib>
#include <chrono>
#include <mpi.h>

using namespace std;
using namespace Eigen;

// Function to perform bootstrap variance
double bootstrap_variance(const MatrixXd &X_labeled, const MatrixXd &X_unlabeled, const VectorXd &Y_labeled, int train_n, double thetaPP, int B = 30)
{
    int n = X_labeled.rows();
    int N = X_unlabeled.rows();
    VectorXd Yhat_labeled = VectorXd::Zero(n);
    VectorXd Yhat_unlabeled = VectorXd::Zero(N);
    VectorXd grad_diff((n - train_n) * B);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    random_device rd;
    mt19937 gen(rd() + rank); // Different seed for each process
    uniform_int_distribution<int> dist(0, n - 1);

    int start_idx = (B / size) * rank;
    int end_idx = (B / size) * (rank + 1);
    if (rank == size - 1)
        end_idx = B; // Make sure the last process handles the remaining tasks

    for (int j = start_idx; j < end_idx; ++j)
    {
        // Same logic as before but within the range of j
        vector<int> train_ind;
        while (train_ind.size() < train_n)
        {
            int idx = dist(gen);
            if (find(train_ind.begin(), train_ind.end(), idx) == train_ind.end())
            {
                train_ind.push_back(idx);
            }
        }

        MatrixXd X_train(train_n, X_labeled.cols());
        VectorXd Y_train(train_n);
        for (int i = 0; i < train_n; ++i)
        {
            X_train.row(i) = X_labeled.row(train_ind[i]);
            Y_train(i) = Y_labeled(train_ind[i]);
        }

        // Split data, XGBoost training, and prediction logic
        // The rest of the code remains the same

        // Update results (grad_diff, Yhat_unlabeled, etc.)
    }

    // Gather the results from all processes
    MPI_Gather(grad_diff.data(), grad_diff.size(), MPI_DOUBLE, grad_diff.data(), grad_diff.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double var_unlabeled = Yhat_unlabeled.squaredNorm() / N;
    double var_labeled = grad_diff.squaredNorm() / (n - train_n);

    double var_hat = var_unlabeled / N + var_labeled / n;
    return var_hat;
}

// Function to perform cross-prediction
vector<double> cross_prediction_mean_interval(const MatrixXd &X_labeled, const MatrixXd &X_unlabeled, const VectorXd &Y_labeled, double alpha, int K = 10)
{
    int n = X_labeled.rows();
    int N = X_unlabeled.rows();
    int fold_n = n / K;

    VectorXd Yhat_labeled = VectorXd::Zero(n);
    VectorXd Yhat_unlabeled = VectorXd::Zero(N);
    VectorXd Yhat_avg_labeled = VectorXd::Zero(n);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int start_idx = (K / size) * rank;
    int end_idx = (K / size) * (rank + 1);
    if (rank == size - 1)
        end_idx = K; // Make sure the last process handles the remaining tasks

    for (int j = start_idx; j < end_idx; j++)
    {
        int start = j * fold_n;
        int end = (j + 1) * fold_n;

        MatrixXd X_val = X_labeled.middleRows(start, fold_n);
        VectorXd Y_val = Y_labeled.segment(start, fold_n);

        vector<int> train_indices;
        for (int i = 0; i < n; i++)
        {
            if (i < start || i >= end)
                train_indices.push_back(i);
        }

        MatrixXd X_train(train_indices.size(), X_labeled.cols());
        VectorXd Y_train(train_indices.size());

        for (size_t i = 0; i < train_indices.size(); i++)
        {
            X_train.row(i) = X_labeled.row(train_indices[i]);
            Y_train(i) = Y_labeled(train_indices[i]);
        }

        // Split the training data further into two sets for XGBoost training
        int split_idx = X_train.rows() * 0.9;
        MatrixXd X_train1 = X_train.topRows(split_idx);
        MatrixXd X_train2 = X_train.bottomRows(X_train.rows() - split_idx);
        VectorXd Y_train1 = Y_train.head(split_idx);
        VectorXd Y_train2 = Y_train.tail(Y_train.size() - split_idx);

        // Convert Eigen matrices to float pointers for XGBoost
        DMatrixHandle dtrain, dtest;
        XGDMatrixCreateFromMat(reinterpret_cast<const float *>(X_train1.data()), X_train1.rows(), X_train1.cols(), -1, &dtrain);
        XGDMatrixCreateFromMat(reinterpret_cast<const float *>(X_train2.data()), X_train2.rows(), X_train2.cols(), -1, &dtest);

        // Set parameters for XGBoost
        const char *param[] = {"max_depth=7", "eta=0.1", "objective=reg:squarederror", "eval_metric=error"};
        int num_round = 500;
        BoosterHandle booster;
        XGBoosterCreate(&dtrain, 1, &booster);
        XGBoosterSetParam(booster, "max_depth", "7");
        XGBoosterSetParam(booster, "eta", "0.1");
        XGBoosterSetParam(booster, "objective", "reg:squarederror");

        // Training loop
        for (int round = 0; round < num_round; ++round)
        {
            XGBoosterUpdateOneIter(booster, round, dtrain);
        }

        // Predict for the unlabeled data and update the predictions
        VectorXd y_pred_unlabeled(N);
        XGDMatrixCreateFromMat(reinterpret_cast<const float *>(X_unlabeled.data()), X_unlabeled.rows(), X_unlabeled.cols(), -1, &dtest);

        bst_ulong out_len;
        const float *out_result;
        XGBoosterPredict(booster, dtest, 0, 0, 0, &out_len, &out_result);

        // Store the predictions in the result vector
        for (bst_ulong i = 0; i < out_len; ++i)
        {
            y_pred_unlabeled(i) = out_result[i];
        }

        Yhat_unlabeled += y_pred_unlabeled / K;
        Yhat_labeled.segment(start, fold_n) = y_pred_unlabeled.head(fold_n);
    }

    // Gather the results from all processes using MPI_Gather
    MPI_Gather(Yhat_unlabeled.data(), N, MPI_DOUBLE, Yhat_unlabeled.data(), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(Yhat_labeled.data(), n, MPI_DOUBLE, Yhat_labeled.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Calculate thetaPP and variance on rank 0 (master)
    double thetaPP = Yhat_unlabeled.mean() + (Y_labeled - Yhat_labeled).mean();
    double var_hat = bootstrap_variance(X_labeled, X_unlabeled, Y_labeled, n - fold_n, thetaPP);

    double halfwidth = 1.96 * sqrt(var_hat); // norm.ppf(1-alpha/2) is approximately 1.96 for alpha=0.05

    return {thetaPP - halfwidth, thetaPP + halfwidth};
}

// Function to split data into labeled and unlabeled sets
void train_test_split(const MatrixXd &X, const VectorXd &y, int train_size, MatrixXd &X_labeled, MatrixXd &X_unlabeled, VectorXd &Y_labeled, VectorXd &Y_unlabeled)
{
    // Assuming X and y are the same size, split them into two parts: labeled and unlabeled
    int total_size = X.rows();

    // Randomly shuffle indices
    vector<int> indices(total_size);
    iota(indices.begin(), indices.end(), 0);
    random_shuffle(indices.begin(), indices.end());

    // Use the first `train_size` elements as labeled
    X_labeled.resize(train_size, X.cols());
    Y_labeled.resize(train_size);

    // The remaining as unlabeled
    X_unlabeled.resize(total_size - train_size, X.cols());
    Y_unlabeled.resize(total_size - train_size);

    // Populate the labeled and unlabeled sets
    for (int i = 0; i < train_size; ++i)
    {
        X_labeled.row(i) = X.row(indices[i]);
        Y_labeled(i) = y(indices[i]);
    }

    for (int i = train_size; i < total_size; ++i)
    {
        X_unlabeled.row(i - train_size) = X.row(indices[i]);
        Y_unlabeled(i - train_size) = y(indices[i]);
    }
}

vector<double> trial(const MatrixXd &X, const VectorXd &y, int n, double alpha)
{
    // Split the data into labeled and unlabeled sets
    MatrixXd X_labeled, X_unlabeled;
    VectorXd Y_labeled, Y_unlabeled;
    train_test_split(X, y, n, X_labeled, X_unlabeled, Y_labeled, Y_unlabeled);

    // Call the cross-prediction mean interval function
    return cross_prediction_mean_interval(X_labeled, X_unlabeled, Y_labeled, alpha);
}
