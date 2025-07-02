#ifndef TENSOR_H
#define TENSOR_H

#include <cstdio>
#include <iostream>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace NN {
    class BaseModule;
}

class Tensor;

struct gradNode {
    std::vector<gradNode*> children;
    NN::BaseModule* module;

    gradNode() : module(nullptr) {}
};

class Tensor {
public:

    Tensor() : owner(false), grad_(false), data_(nullptr), size_(0) {}
    Tensor(std::vector<int> dims, bool grad = false, std::string preset = "none");
    Tensor(std::vector<int> dims, float* data, bool copy = true, bool grad = true);
    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);
    ~Tensor();


    Tensor operator[](int index) const;
    float toFloat() const;
    Tensor operator*(float f) const;

    void operator=(const float& f);
    void operator=(const std::vector<std::vector<float>>& v);

    Tensor transpose();
    Tensor matmul(const Tensor& t) const;
    Tensor add(const Tensor& t) const;
    float sum() const;
    Tensor operator^(int power) const;
    
    static Tensor fromList(const std::vector<float>& data, std::vector<int> dims);
    std::vector<int> dims() const;
    std::vector<int> strides() const;
    int size() const;

    void print() const;
    pybind11::object toList() const;

    bool owner;
    bool grad_;
    gradNode root_;
    float* data() const;

private:
    void printRecursive(float*, const std::vector<int>&, int, int) const;
    std::vector<int> dims_;
    float* data_;
    int size_;
    std::vector<int> strides_;
};



#endif
