#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>

void loadDataFromFile(const std::string& filename, std::vector<double>& data) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        double value;
        while (ss >> value) {
            data.push_back(value);
            if (ss.peek() == ',') {
                ss.ignore();
            }
        }
    }

    file.close();
}

void loadData(std::vector<double>& Y, std::vector<double>& Y_hat, std::vector<double>& Y_hat_unlabeled) {
    loadDataFromFile("data/Y.csv", Y);
    loadDataFromFile("data/Y_hat.csv", Y_hat);
    loadDataFromFile("data/Y_hat_unlabeled.csv", Y_hat_unlabeled);

    std::cout << "Data loaded" << std::endl;
    std::cout << "Y size: " << Y.size() << std::endl;
    std::cout << "Y_hat size: " << Y_hat.size() << std::endl;
    std::cout << "Y_hat_unlabeled size: " << Y_hat_unlabeled.size() << std::endl;
}