#include <iostream>
#include <fstream>
#include <filesystem>
#include <gdal_priv.h>
#include <cpl_conv.h>
#include <vector>
#include <string>
#include <cmath>

namespace fs = std::filesystem;

void write_to_csv(const std::vector<std::string> &row_data, const std::string &csv_filename)
{
    std::ofstream csv_file(csv_filename, std::ios::app);
    if (csv_file.is_open())
    {
        for (size_t i = 0; i < row_data.size(); i++)
        {
            csv_file << row_data[i];
            if (i < row_data.size() - 1)
            {
                csv_file << ","; // Add comma between columns
            }
        }
        csv_file << "\n"; // New line after each row
        csv_file.close();
    }
    else
    {
        std::cerr << "Failed to open CSV file: " << csv_filename << std::endl;
    }
}

// 计算图像块的平均值
double calculate_block_mean(GDALRasterBand *band, int block_size)
{
    int width = band->GetXSize();
    int height = band->GetYSize();

    double sum = 0.0;
    int count = 0;

    for (int y = 0; y < height; y += block_size)
    {
        for (int x = 0; x < width; x += block_size)
        {
            int block_width = std::min(block_size, width - x);
            int block_height = std::min(block_size, height - y);
            std::vector<float> data(block_width * block_height);

            // 读取块内的像素值
            band->RasterIO(GF_Read, x, y, block_width, block_height, data.data(), block_width, block_height, GDT_Float32, 0, 0);

            // 计算块的平均值
            for (int i = 0; i < block_width * block_height; ++i)
            {
                sum += data[i];
                count++;
            }
        }
    }

    return (count > 0) ? sum / count : 0.0;
}

// 计算波段像素值和的方差
double calculate_band_sum_variance(GDALRasterBand *band)
{
    int width = band->GetXSize();
    int height = band->GetYSize();

    double sum = 0.0;
    std::vector<double> band_sums;

    // 计算每一行的和
    for (int y = 0; y < height; ++y)
    {
        std::vector<float> row_data(width);
        band->RasterIO(GF_Read, 0, y, width, 1, row_data.data(), width, 1, GDT_Float32, 0, 0);
        double row_sum = 0.0;
        for (int x = 0; x < width; ++x)
        {
            row_sum += row_data[x];
        }
        band_sums.push_back(row_sum);
    }

    // 计算加和方差
    double mean_sum = 0.0;
    for (double row_sum : band_sums)
    {
        mean_sum += row_sum;
    }
    mean_sum /= band_sums.size();

    double variance = 0.0;
    for (double row_sum : band_sums)
    {
        variance += (row_sum - mean_sum) * (row_sum - mean_sum);
    }

    return (band_sums.size() > 1) ? variance / (band_sums.size() - 1) : 0.0;
}

void extract_image_features(const std::string &filepath, const std::string &csv_filename)
{
    GDALAllRegister(); // 初始化 GDAL 库

    // 打开 TIF 文件
    GDALDataset *dataset = (GDALDataset *)GDALOpen(filepath.c_str(), GA_ReadOnly);
    if (dataset == nullptr)
    {
        std::cerr << "Failed to open file: " << filepath << std::endl;
        return;
    }

    // 获取影像的宽度和高度
    int width = dataset->GetRasterXSize();
    int height = dataset->GetRasterYSize();

    // 获取波段数量
    int bandCount = dataset->GetRasterCount();
    std::cout << "File: " << filepath << ", Width: " << width << ", Height: " << height << ", Bands: " << bandCount << std::endl;

    // 准备要写入 CSV 的特征数据
    std::vector<std::string> row_data;
    row_data.push_back(filepath);                  // 将文件路径作为第一列数据
    row_data.push_back(std::to_string(width));     // 图像宽度
    row_data.push_back(std::to_string(height));    // 图像高度
    row_data.push_back(std::to_string(bandCount)); // 波段数量

    // 计算每个波段的统计特征（最小值、最大值、均值、标准差）
    for (int i = 1; i <= 6; i++) // 假设最多有6个波段
    {
        GDALRasterBand *band = nullptr;
        if (i <= bandCount)
        {
            band = dataset->GetRasterBand(i);
        }

        double min = 0, max = 0, mean = 0, stdDev = 0;
        int success = 0;
        if (band)
        {
            min = band->GetMinimum(&success);
            max = band->GetMaximum(&success);
            band->GetStatistics(true, true, &min, &max, &mean, &stdDev);
        }

        // 将特征数据加入行数据
        row_data.push_back(std::to_string(min));    // Min
        row_data.push_back(std::to_string(max));    // Max
        row_data.push_back(std::to_string(mean));   // Mean
        row_data.push_back(std::to_string(stdDev)); // StdDev

        // 计算并添加 block mean 和 band sum variance
        if (band)
        {
            double block_mean = calculate_block_mean(band, 10); // 10x10 块大小
            double band_sum_variance = calculate_band_sum_variance(band);
            row_data.push_back(std::to_string(block_mean));        // Block Mean
            row_data.push_back(std::to_string(band_sum_variance)); // Band Sum Variance
        }
        else
        {
            row_data.push_back(""); // Block Mean (if band doesn't exist)
            row_data.push_back(""); // Band Sum Variance (if band doesn't exist)
        }
    }

    // 如果波段数少于6，填充空值
    for (int i = bandCount + 1; i <= 6; i++)
    {
        row_data.push_back(""); // Min
        row_data.push_back(""); // Max
        row_data.push_back(""); // Mean
        row_data.push_back(""); // StdDev
        row_data.push_back(""); // Block Mean
        row_data.push_back(""); // Band Sum Variance
    }

    // 确保数据列数和标题列数一致
    std::cout << "Row has " << row_data.size() << " columns." << std::endl;

    // 写入 CSV 文件
    write_to_csv(row_data, csv_filename);

    // 释放资源
    GDALClose(dataset);
}

void process_directory(const std::string &directory, const std::string &csv_filename)
{
    // 写入 CSV 文件的标题行
    std::vector<std::string> header = {"Filepath", "Width", "Height", "BandCount",
                                       "Band1_Min", "Band1_Max", "Band1_Mean", "Band1_StdDev", "Band1_BlockMean", "Band1_BandSumVariance",
                                       "Band2_Min", "Band2_Max", "Band2_Mean", "Band2_StdDev", "Band2_BlockMean", "Band2_BandSumVariance",
                                       "Band3_Min", "Band3_Max", "Band3_Mean", "Band3_StdDev", "Band3_BlockMean", "Band3_BandSumVariance",
                                       "Band4_Min", "Band4_Max", "Band4_Mean", "Band4_StdDev", "Band4_BlockMean", "Band4_BandSumVariance",
                                       "Band5_Min", "Band5_Max", "Band5_Mean", "Band5_StdDev", "Band5_BlockMean", "Band5_BandSumVariance",
                                       "Band6_Min", "Band6_Max", "Band6_Mean", "Band6_StdDev", "Band6_BlockMean", "Band6_BandSumVariance"};
    write_to_csv(header, csv_filename);

    // 遍历文件夹及子文件夹
    for (const auto &entry : fs::recursive_directory_iterator(directory))
    {
        if (entry.is_regular_file() && entry.path().extension() == ".tif")
        {
            extract_image_features(entry.path().string(), csv_filename);
        }
    }
}

int main()
{
    std::string directory = "./sentinel/dataset";    // 需要遍历的文件夹路径
    std::string csv_filename = "./data/image_features.csv"; // 输出的 CSV 文件路径

    // 处理文件夹并将特征保存到 CSV 文件
    process_directory(directory, csv_filename);

    std::cout << "Feature extraction completed and saved to " << csv_filename << std::endl;
    return 0;
}
