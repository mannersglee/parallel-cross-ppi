g++ -o omp-mean omp-mean.cpp -std=c++11 -I/usr/include/eigen3 -I/path/to/xgboost/include -L/path/to/xgboost/lib -lxgboost -fopenmp
./omp-mean