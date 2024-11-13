mpic++ -o parallel-mean parallel-mean.cpp -std=c++11 -I/usr/include/eigen3 -I/path/to/xgboost/include -L/path/to/xgboost/lib -lxgboost
mpirun -np 4 ./parallel-mean
