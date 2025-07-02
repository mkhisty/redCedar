#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <pybind11/pytypes.h>
#include <variant>
#include <vector>
#include "tensor.hpp"
#include <stdexcept>
#include <algorithm>
#include <pybind11/stl.h>
#include "nn.hpp"
#include <random>
#include <cmath>

Tensor::Tensor(std::vector<int> dims,bool grad,std::string preset){
    owner = true;
    dims_ = dims;
    size_ =1;
    grad_ = grad;
    if(grad_){
        root_ =  gradNode();
    }
    std::reverse(dims.begin(),dims.end());
    int f=1;
    for(int i:dims){
       size_*=i;
       strides_.push_back(f);
       f*=i;
    }
    std::reverse(strides_.begin(),strides_.end());

    data_ = (float*)(malloc(size_*sizeof(float)));
    if(data_ == nullptr){
        throw std::runtime_error("Get more RAM");
    }
    std::memset(data_, 0, size_*sizeof(float));
    if(preset=="identity"){
        if(dims_.size()!=2){
            throw std::runtime_error("only for 2d");
        }
        std::memset(data_, 0, size_*sizeof(float));
        for(int i=0;i<dims_[0];i++){
            data_[i*dims_[1]+i] = 1;
        }
    }else if(preset=="random"){

        std::random_device rd;
        std::mt19937 gen(rd());


        float limit = std::sqrt(6.0f / (dims_[0] + dims[1]));
        std::uniform_real_distribution<float> dis(-limit, limit);

        for(size_t i = 0; i < size_; ++i) {
            data_[i] = dis(gen);
        }
    }

}


Tensor::Tensor(const Tensor& other) {
    dims_ = other.dims_;
    size_ = other.size_;
    strides_ = other.strides_;
    grad_ = other.grad_;
    root_ = other.root_;  
    owner = true; 

    if (other.data_ != nullptr && size_ > 0) {
        data_ = (float*)malloc(size_ * sizeof(float));
        if (data_ == nullptr) {
            throw std::runtime_error("Get more ram");
        }
        memcpy(data_, other.data_, size_ * sizeof(float));
    } else {
        data_ = nullptr;
    }
}

Tensor::Tensor(std::vector<int> dims, float* data,bool copy,bool grad){
    dims_ = dims;
    size_ =1;
    owner = copy;
    grad_ = grad;
    if(dims.empty()){
        strides_.push_back(0);
    }

    std::reverse(dims.begin(),dims.end());
    int f=1;
    for(int i:dims){
       size_*=i;
       strides_.push_back(f);
       f*=i;
    }
    std::reverse(strides_.begin(),strides_.end());

    if(copy){
    data_ = (float*)(malloc(size_*sizeof(float)));
    if(data_ == nullptr){
        throw std::runtime_error("Get more RAM");
    }
    memcpy(data_, data, size_*sizeof(float));
    }else{
        data_ = data;
    }
}

Tensor::~Tensor() {
    if (owner && data_ != nullptr) {
        free(data_);
        data_ = nullptr;
    }
}



Tensor Tensor::operator[](int index) const{
    if(index<0 || index>=dims_[0]){
        throw std::out_of_range("Index out of range");
    }
    if(dims_.begin() == dims_.end()){
        throw std::runtime_error("# of indices># of dims");
    }
    float* ptr = data_ + (index * strides_[0]);
    bool copy = false;
    return Tensor(std::vector<int>(dims_.begin()+1,dims_.end()), ptr,copy);
}


Tensor Tensor::transpose() {
    if (dims_.size() != 2) {
        throw std::runtime_error("2d only");
    }

    float* new_data = (float*)malloc(sizeof(float) * size_);
    if (!new_data) {
        throw std::runtime_error("Get more RAM");
    }

    for (int i = 0; i < dims_[0]; i++) {
        for (int j = 0; j < dims_[1]; j++) {
            int old_idx = i * strides_[0] + j * strides_[1];
            int new_idx = j * dims_[0] + i;
            new_data[new_idx] = data_[old_idx];
        }
    }

    data_ = new_data;
    owner = true;

    return Tensor({dims_[1],dims_[0]},new_data,true);
}


Tensor Tensor::operator*(float f) const{
    float* newData = (float*)malloc(size_*sizeof(float));

    std::memcpy(newData, data_, size_*sizeof(float));
    for (int i = 0; i < size_; i++) {
        newData[i] *= f;
    }
    return Tensor({dims_[0],dims_[1]},newData,true);
}

Tensor Tensor::matmul(const Tensor &t) const{

    std::vector<int> dims = {dims_[0],t.dims()[1]};
    float* data = (float*)(malloc(dims[0]*dims[1]*sizeof(float)));
    if(data == nullptr){
        throw std::runtime_error("Get more RAM");
    }
    std::memset(data, 0, dims[0]*dims[1]*sizeof(float));
    for(int i=0;i<dims_[0];i++){
        for(int j=0;j<t.dims()[1];j++){
            for(int k=0;k<dims_[1];k++){
                data[i*dims[1]+j] += data_[i*dims_[1]+k] * t.data_[k*t.dims()[1]+j];
            }
        }
    }
    Tensor result(dims, data);

    return result;
}


Tensor Tensor::add(const Tensor& t) const{
    if(dims_ != t.dims_){
        throw std::runtime_error("Dimensions do not match");
    }
    float* new_data = (float*)(malloc(size_*sizeof(float)));
    if(new_data == nullptr){
        throw std::runtime_error("Get more RAM");
    }
    std::memcpy(new_data, data_, size_*sizeof(float));
    for(size_t i=0;i<size_;i++){
        new_data[i] += t.data_[i];
    }
    Tensor result(dims_, new_data);

    return result;
}


void Tensor::operator=(const float &f){
    if(!dims_.empty() && dims_[0]!=1){
        throw std::runtime_error("Tensor is not scalar");
    }
    data_[0] = f;
}

void Tensor::operator=(const std::vector<std::vector<float>> &v){
    if(static_cast<int>(dims_.size()) !=2){
        throw std::runtime_error("Only 2d tensors can be set through =");
    }
    if(static_cast<int>(v.size()) != dims_[0] || static_cast<int>(v[0].size()) != dims_[1]){
        throw std::runtime_error("Vector size does not match");
    }
    for(size_t i=0;i<v.size();i++){
        for(size_t j=0;j<v[0].size();j++){
            data_[i*dims_[1]+j] = v[i][j];
        }
    }
}
Tensor& Tensor::operator=(const Tensor& other) {
    if (this == &other) {
        return *this;
    }

    if (dims_ != other.dims_) {
        if (owner && data_) {
            delete[] data_;
        }

        dims_ = other.dims_;
        size_ = 1;
        for (int dim : dims_) {
            size_ *= dim;
        }

        strides_.resize(dims_.size());
        strides_.back() = 1;
        for (int i = dims_.size() - 2; i >= 0; --i) {
            strides_[i] = strides_[i + 1] * dims_[i + 1];
        }

        data_ = new float[size_];
        owner = true;
    }

    std::memcpy(data_, other.data_, size_ * sizeof(float));

    grad_ = other.grad_;
    root_ = other.root_;
    return *this;
}
float Tensor::toFloat() const{
    if (size_ == 1 && data_ != nullptr) {
        return data_[0];
    } else {
        throw std::runtime_error("Tensor is not scalar or size 1");
    }
}


void Tensor::print() const {
    if (dims_.empty()) {
        std::cout << "[]" << std::endl;
        return;
    }
    std::cout<<"Dims: ";
    for(int i=0;i<dims_.size();i++){
        std::cout<<dims_[i]<<" ";
    }
    std::cout<<std::endl;
    printRecursive(data_, dims_, 0, 0);
    std::cout << std::endl;
}

void Tensor::printRecursive(float* data, const std::vector<int>& dims,
                           int dim_index, int offset) const {
    if (dim_index == dims.size() - 1) {
        std::cout << "[";
        for (int i = 0; i < dims[dim_index]; ++i) {
            std::cout << data[offset + i];
            if (i < dims[dim_index] - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]";
        return;
    }

    std::cout << "[";
    int stride = 1;
    for (int i = dim_index + 1; i < dims.size(); ++i) {
        stride *= dims[i];
    }

    for (int i = 0; i < dims[dim_index]; ++i) {
        printRecursive(data, dims, dim_index + 1, offset + i * stride);
        if (i < dims[dim_index] - 1) {
            std::cout << ", ";
            if (dim_index == 0 && dims.size() > 2) {
                std::cout << "\n ";
            }
        }
    }
    std::cout << "]";
}



pybind11::object Tensor::toList() const {
    if (dims_.size() == 1) {
        pybind11::list list;
        for (auto i = 0; i < dims_[0]; i++) {
            list.append(pybind11::cast(static_cast<float>(data_[i * strides_[0]])));
        }
        return list;
    } else {
        pybind11::list list;
        for (auto i = 0; i < dims_[0]; i++) {

            list.append((*this)[i].toList());

        }
        return list;
    }
}


std::vector<int> Tensor::dims() const{
    return dims_;
}
std::vector<int> Tensor::strides() const{
    return strides_;
}



float Tensor::sum() const {
    float sum = 0.0f;
    for (auto i = 0; i < size(); i++) {
        sum += data_[i];
    }
    return sum;
}

Tensor Tensor::operator^(int power) const {
    Tensor result(dims_, false, "none");
    for (auto i = 0; i < size(); i++) {
        result.data_[i] = std::pow(data_[i], power);
    }
    return result;
}

int Tensor::size() const {
    return size_;
}


float* Tensor::data() const {
    return data_;
}

Tensor Tensor::fromList(const std::vector<float>& data, std::vector<int> dims) {
    int total_size = 1;
    for (int dim : dims) {
        total_size *= dim;
    }

    if (static_cast<int>(data.size()) != total_size) {
        throw std::runtime_error("Data size does not match tensor dimensions");
    }

    float* tensor_data = (float*)malloc(total_size * sizeof(float));
    if (tensor_data == nullptr) {
        throw std::runtime_error("Get more RAM");
    }

    for (int i = 0; i < total_size; i++) {
        tensor_data[i] = data[i];
    }

    return Tensor(dims, tensor_data, true, false);
}
