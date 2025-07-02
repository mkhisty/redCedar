#!/bin/bash

# Build script for the tensor neural network module

echo "Building tensor module..."

# Create build directory if it doesn't exist
mkdir -p build
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake ..

# Build the module
echo "Compiling..."
make -j$(nproc)

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "Build successful!"

    # Copy the built module to the parent directory for easy access
    if [ -f tensor*.so ]; then
        cp tensor*.so ../
        echo "Module copied to parent directory"
    fi

    echo "You can now run: python test/train_network.py"
else
    echo "Build failed!"
    exit 1
fi
