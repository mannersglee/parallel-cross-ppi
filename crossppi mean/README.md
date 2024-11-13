- 运行 ``bash install-sgboost``，即可安装配置成功xgboost，其他库和依赖自行apt install
- 保证 xgboost 文件夹与代码 mean.cpp，mean.hpp 等在同一个路径下 ``bash mean.sh`` 即可计算均值区间估计
- mean.cpp 主函数中这部分代码用于控制测试计算量大小，自行调整

```cpp
Parameters
int N = 10000;                                                        // Size of unlabeled data
vector<int> ns = {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000}; // Sizes of labeled data
int num_trials = 100;
double alpha = 0.1;

// Data generating parameters
int d = 2;
vector<double> Rsqs = {0, 0.5, 1};
double var_y = 4;
double mu = 4;
VectorXd beta = VectorXd::Zero(d);
```

- parallel-mean.cpp 还未实现
