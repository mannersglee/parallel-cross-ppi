sudo apt-get update
sudo apt-get install build-essential cmake
sudo apt-get install libopenblas-dev
sudo apt-get install libboost-all-dev
sudo apt-get install libomp-dev
git clone --recursive https://github.com/dmlc/xgboost
cd xgboost
mkdir build
cd build
cmake ..
make -j$(nproc)
sudo make install