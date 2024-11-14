rm image_features.csv 
g++ -std=c++17 -o extract_features extract_features.cpp -lstdc++fs -lgdal
./extract_features
rm extract_features