# Compiler and Flags
CXX = mpic++
CXXFLAGS = -std=c++17 -O2
LIBS = -lboost_math_c99

# Source files and output executable
SRC = main.cpp
OUT = main

# Default target: build the executable
all: $(OUT)

$(OUT): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(OUT) $(SRC) $(LIBS)

# Clean up the compiled output
clean:
	rm -f $(OUT)