# Compiler and flags
CXX = mpic++
CXXFLAGS = -O2

# Target executable
TARGET = main

# Source files
SRC = main.cpp

# Build the target
$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

# Run the program with mpiexec
run: $(TARGET)
	mpiexec -np 4 ./$(TARGET)

# Clean up build files
clean:
	rm -f $(TARGET)
