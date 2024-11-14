#include "parallel-mean.hpp"

using namespace std;
using namespace Eigen;

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv); // Initialize MPI
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get the rank of the current process
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get the number of processes

    // Parameters
    int N = 1000;           // Size of unlabeled data
    vector<int> ns = {100}; // Sizes of labeled data
    int num_trials = 30;
    double alpha = 0.1;

    // Data generating parameters
    int d = 2;
    vector<double> Rsqs = {0, 0.5, 1};
    double var_y = 4;
    double mu = 4;
    VectorXd beta = VectorXd::Zero(d);

    // Random number generator
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<double> dist(0.0, 1.0);

    // 记录程序开始的时间
    auto start_time = std::chrono::high_resolution_clock::now();

    // Loop over R^2 values (each process will handle a subset of Rsqs)
    for (size_t rsq_idx = rank; rsq_idx < Rsqs.size(); rsq_idx += size)
    {
        double Rsq = Rsqs[rsq_idx];

        // Prepare file name for saving results
        stringstream ss;
        ss.str("");
        ss << "mean_results/Rsq_" << std::fixed << std::setprecision(2) << Rsq << ".csv";
        string filename = ss.str();

        // Skip if result file exists
        if (ifstream(filename))
            continue;

        beta.setZero();                                   // Reset beta
        beta = sqrt(Rsq * var_y / d) * VectorXd::Ones(d); // Adjust beta according to Rsq

        vector<vector<string>> result_for_Rsq;

        // Loop over different n values (sizes of labeled data)
        for (size_t n_idx = 0; n_idx < ns.size(); ++n_idx)
        {
            int n = ns[n_idx];
            vector<vector<string>> result_for_n;

            // Run multiple trials
            for (int i = rank; i < num_trials; i += size)
            { // Each process runs different trials
                // Generate data
                MatrixXd X(n + N, d);
                VectorXd y(n + N);

                // Generate feature matrix X and outcome y
                for (int j = 0; j < (n + N); ++j)
                {
                    for (int k = 0; k < d; ++k)
                    {
                        X(j, k) = dist(gen);
                    }
                    y(j) = X.row(j).dot(beta) + sqrt(var_y * (1 - Rsq)) * dist(gen) + mu;
                }

                // Call the trial function to get confidence intervals and PPI
                vector<double> trial_result = trial(X, y, n, alpha);
                double cppi_lower = trial_result[0];
                double cppi_upper = trial_result[1];

                bool cppi_coverage = (cppi_lower <= mu) && (mu <= cppi_upper);

                // Store results for this trial
                result_for_n.push_back({to_string(cppi_lower),
                                        to_string(cppi_upper),
                                        to_string(cppi_coverage),
                                        "cross-prediction",
                                        to_string(n),
                                        to_string(Rsq)});
            }

            // Store results for this n
            result_for_Rsq.insert(result_for_Rsq.end(), result_for_n.begin(), result_for_n.end());
        }

        // Using MPI_Allgather to collect results from all processes
        vector<vector<string>> all_results;

        // Assuming each process stores its results in result_for_Rsq
        // Gather all results into all_results (every process gets the full result set)
        int total_results = result_for_Rsq.size();
        vector<int> recv_counts(size, total_results);
        vector<int> displs(size, 0);

        // Prepare a buffer to store results from all processes
        vector<vector<string>> gathered_results;

        // Gather the results from all processes
        MPI_Allgather(&result_for_Rsq[0], total_results, MPI_CHAR, &gathered_results[0], total_results, MPI_CHAR, MPI_COMM_WORLD);

        // Write results to CSV file
        if (rank == 0)
        { // Only process 0 writes the file
            ofstream outfile(filename);
            outfile << "lb,ub,coverage,estimator,n,R^2\n";
            for (const auto &row : gathered_results)
            {
                for (size_t i = 0; i < row.size(); ++i)
                {
                    outfile << row[i];
                    if (i != row.size() - 1)
                        outfile << ",";
                }
                outfile << "\n";
            }
            outfile.close();
        }

        // Output progress to rank 0
        if (rank == 0)
        {
            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = end_time - start_time;
            std::cout << "Progress for Rsq = " << Rsq << " | Time elapsed: " << duration.count() << " seconds\n";
        }
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}
