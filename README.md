# parallel-cross-ppi

目前实现：

- cross-ppi均值区间估计（MPI并行）
- cross-ppi分位数区间估计（MPI并行）

## 环境

- 操作系统：Linux Ubuntu (推荐最新的 LTS 版本)
- 编译器：gcc、g++、gfortran
- 构建工具：CMake

在源码编译之前，确保系统中已经安装了 `gcc`、`g++`、`cmake` 和 `gfortran`。检查方法如下：

```bash
gcc --version
g++ --version
cmake --version
gfortran --version
```

安装`boost`库：
```bash
sudo apt-get update
sudo apt-get install libboost-all-dev
```
验证安装是否成功：
```bash
dpkg -s libboost-all-dev | grep "Status"
```

安装`mpi`:
```bash
sudo apt-get update
sudo apt-get install mpich
```
验证安装是否成功：
```bash
mpiexec --version
```

## 运行

```bash
./analysis.sh
```