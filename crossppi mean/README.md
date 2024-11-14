## 运行方法

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


## 运行结果

| 方法       | 运行时间  | 提升    |
| ---------- | --------- | ------- |
| 串行       | 942.473秒 |         |
| OpenMP并行 | 541.476秒 | 42.547% |

### 串行

串行运行少量数据

![image-20241114010621938](img\image-20241114010621938.png)

运行时间942.473秒

运行全部数据（和论文代码相同数据）

![image-20241114074529659](img\image-20241114074529659.png)

运行时间太长，进程被杀

### OpenMP并行加速

在这段代码中，使用 OpenMP 并行加速的部分主要集中在两个地方：

1. **外层 `Rsq` 循环的并行化**：
   
   ```cpp
   #pragma omp parallel for
   for (size_t rsq_idx = 0; rsq_idx < Rsqs.size(); ++rsq_idx)
   ```
   这个循环遍历不同的 `Rsq` 值。每个 `Rsq` 的计算是独立的，因此可以在多个线程中并行执行。每个线程处理一个 `Rsq` 的结果计算，减少了该部分的计算时间。
   
2. **内层 `num_trials` 循环的并行化**：
   ```cpp
   #pragma omp parallel for
   for (int i = 0; i < num_trials; ++i)
   ```
   这个循环用于执行多个试验 (`num_trials`)，每个试验是独立的，因此也可以在多个线程中并行执行。每个线程处理一个试验的结果计算，进一步加速整体运行。

并行化加速的具体效果：

- **`Rsq` 循环并行化**：当有多个不同的 `Rsq` 值时（例如，`Rsqs = {0, 0.5, 1}`），每个 `Rsq` 对应的计算是独立的，使用并行化后，多个 `Rsq` 值的计算可以同时进行，显著减少了计算时间。
  
- **`num_trials` 循环并行化**：每个 `trial` 的计算也是独立的（即每次生成数据并调用 `trial()` 函数），通过并行化，多个试验可以在不同线程中并行执行，进一步减少了总的运行时间。

关键点：

- **数据生成**：数据的生成部分（矩阵 `X` 和向量 `y`）是每个试验独立生成的，因此可以并行处理每个试验。
  
- **结果存储**：虽然多个线程并行执行，但是 `result_for_n` 和 `result_for_Rsq` 是共享的，因此使用了 `#pragma omp critical` 来保证对这些结果的更新是线程安全的，避免数据竞争和冲突。

何时看到加速：

- 当 `Rsqs` 和 `num_trials` 的值较大时，OpenMP 的并行化会显著减少运行时间。例如，如果有 10 个 `Rsq` 值和 100 个试验，理论上并行化可以将计算时间缩短到原来的 1/10 到 1/100，具体取决于机器的核心数和 OpenMP 的调度策略。

总结：

通过在 `Rsqs` 和 `num_trials` 的循环中并行化，可以显著加速代码，尤其是在有多个 `Rsq` 和大量试验的情况下，计算效率会有明显提高。

运行截图

![image-20241114185649380](img\image-20241114185649380.png)

运行时间541.476秒
