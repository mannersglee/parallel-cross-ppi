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

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dist(0, n - 1);

    for (int j = 0; j < B; ++j)
    {
        // Randomly select train_n samples
        vector<int> train_ind;
        while (train_ind.size() < train_n)
        {
            int idx = dist(gen);
            if (find(train_ind.begin(), train_ind.end(), idx) == train_ind.end())
            {
                train_ind.push_back(idx);
            }
        }

        // Prepare training data
        MatrixXd X_train(train_n, X_labeled.cols());
        VectorXd Y_train(train_n);
        for (int i = 0; i < train_n; ++i)
        {
            X_train.row(i) = X_labeled.row(train_ind[i]);
            Y_train(i) = Y_labeled(train_ind[i]);
        }

        // Split the training data into X_train1, X_train2 (90% and 10%)
        int split_idx = train_n * 0.9;
        MatrixXd X_train1 = X_train.topRows(split_idx);
        MatrixXd X_train2 = X_train.bottomRows(train_n - split_idx);
        VectorXd Y_train1 = Y_train.head(split_idx);
        VectorXd Y_train2 = Y_train.tail(train_n - split_idx);

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

        // Predict for unlabeled data
        DMatrixHandle dtest_unlabeled;
        XGDMatrixCreateFromMat(reinterpret_cast<const float *>(X_unlabeled.data()), X_unlabeled.rows(), X_unlabeled.cols(), -1, &dtest_unlabeled);
        bst_ulong out_len;
        const float *out_result;
        XGBoosterPredict(booster, dtest_unlabeled, 0, 0, 0, &out_len, &out_result);

        for (bst_ulong i = 0; i < out_len; ++i)
        {
            Yhat_unlabeled(i) += out_result[i] / B;
        }

        // Update the predictions for labeled data
        vector<int> other_inds;
        for (int i = 0; i < n; ++i)
        {
            if (find(train_ind.begin(), train_ind.end(), i) == train_ind.end())
            {
                other_inds.push_back(i);
            }
        }

        MatrixXd X_labeled_other(other_inds.size(), X_labeled.cols());
        VectorXd Y_labeled_other(other_inds.size());
        for (int i = 0; i < other_inds.size(); ++i)
        {
            X_labeled_other.row(i) = X_labeled.row(other_inds[i]);
            Y_labeled_other(i) = Y_labeled(other_inds[i]);
        }

        // Predict for labeled data
        DMatrixHandle dlabeled;
        XGDMatrixCreateFromMat(reinterpret_cast<const float *>(X_labeled_other.data()), X_labeled_other.rows(), X_labeled_other.cols(), -1, &dlabeled);
        const float *labeled_result;
        XGBoosterPredict(booster, dlabeled, 0, 0, 0, &out_len, &labeled_result);

        // Store the gradient differences
        for (int i = 0; i < other_inds.size(); ++i)
        {
            grad_diff(j * (n - train_n) + i) = labeled_result[i] - Y_labeled_other(i);
        }
    }

    // Compute the variances
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

    for (int j = 0; j < K; j++)
    {
        // Split the data into training and validation sets
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

    // Compute thetaPP
    double thetaPP = Yhat_unlabeled.mean() + (Y_labeled - Yhat_labeled).mean();

    // Compute the variance using bootstrap (placeholder function)
    double var_hat = bootstrap_variance(X_labeled, X_unlabeled, Y_labeled, n - fold_n, thetaPP);

    // Calculate half-width for the confidence interval
    double halfwidth = 1.96 * sqrt(var_hat); // norm.ppf(1-alpha/2) is approximately 1.96 for alpha=0.05

    // Return the confidence interval
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
