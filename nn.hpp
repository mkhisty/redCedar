#ifndef NN_H
#define NN_H
#include "tensor.hpp"
#include <iostream>
#include <vector>

class Tensor;
struct gradNode;

namespace NN {
    class BaseModule {
    public:
        BaseModule() {}  
        virtual Tensor forward(Tensor& input) {
            return input;
        }
        virtual Tensor backward(const Tensor& grad, float LR) {
            return grad;
        }
        virtual ~BaseModule() = default;
    };

class Linear : public BaseModule {
public:
    Tensor weights;
    Tensor bias;
    Tensor last_input; 
    Linear(int input_size, int output_size);

    Tensor forward(Tensor& input) override;
    Tensor backward(const Tensor& grad, float LR) override;
};

class MSELoss : public BaseModule{
    public:
    Tensor last_diff; 
    MSELoss() : BaseModule() {}

    Tensor forward(Tensor& input, Tensor& correct) ;
    Tensor backward(const Tensor& grad, float LR) override;
};

class Sigmoid : public BaseModule {
public:
    Tensor last_output;
    Sigmoid() : BaseModule() {}

    Tensor forward(Tensor& input) override;
    Tensor backward(const Tensor& grad, float LR) override;
};


class Model {
    std::vector<BaseModule*> modules;
public:
    Model(std::vector<BaseModule*> modules) : modules(modules) {}

    Tensor forward(Tensor& input){return input;};
    Tensor backward(const Tensor& grad, float LR){return grad;};
};

class Optim {
public:
    float lr;
    Optim(float lr = 0.01) : lr(lr) {}
    void optimize(Tensor& subject);
};



}

#endif
