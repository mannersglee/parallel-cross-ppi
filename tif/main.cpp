#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstring>

#include "serial-cross-ppi.hpp"

int main()
{
    // 记录程序开始的时间
    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<double> Y, Y_hat, Y_hat_unlabeled;
    // loadData(Y, Y_hat, Y_hat_unlabeled, repeat);
    std::ifstream file("data/fire.csv");
    std::string line;

    // 用于存储每一列的数据
    std::vector<std::vector<std::string>> columns_data;

    // 读取列名（第一行）
    if (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string column;

        // 通过读取第一行的列名来确定列数
        while (std::getline(ss, column, ','))
        {
            columns_data.push_back(std::vector<std::string>()); // 初始化每一列的数据容器
        }

        // 读取数据行
        while (std::getline(file, line))
        {
            std::stringstream ss(line);
            std::string value;
            int col_index = 0;

            // 按列分割数据并保存到对应的 vector 中
            while (std::getline(ss, value, ','))
            {
                columns_data[col_index].push_back(value);
                col_index++;
            }
        }

        file.close();

        for (const auto &value : columns_data)
        {

            size_t size = value.size();

            size_t third = size / 3; // 每个部分的大小

            // 获取前 1/3 部分（Y）
            for (size_t i = 0; i < third; ++i)
            {
                Y.push_back(std::stod(value[i])); // 将 std::string 转换为 double
            }

            // 获取中间 1/3 部分（Y_hat）
            for (size_t i = third; i < 2 * third; ++i)
            {
                Y_hat.push_back(std::stod(value[i])); // 将 std::string 转换为 double
            }

            // 获取最后 1/3 部分（Y_hat_unlabeled）
            for (size_t i = 2 * third; i < size; ++i)
            {
                Y_hat_unlabeled.push_back(std::stod(value[i])); // 将 std::string 转换为 double
            }

            int n = 0, N = 0;
            n = Y.size();
            N = Y_hat_unlabeled.size();

            // Compute the cross-ppi confidence interval based on selected method
            std::pair<double, double> ci = serial_cross_ppi_mean_ci(Y,Y_hat,Y_hat_unlabeled,0.0000001);

            std::cout << ci.first << ", " << ci.second << std::endl;
        }
    }

    // 记录程序结束的时间
    auto end_time = std::chrono::high_resolution_clock::now();

    // 计算并输出运行时间
    std::chrono::duration<double> duration = end_time - start_time;
    std::cout << "Program execution time: " << duration.count() << " seconds." << std::endl;

    return 0;
}
