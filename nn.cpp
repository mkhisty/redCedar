#include "nn.hpp"
#include "tensor.hpp"

namespace NN {

    Linear::Linear(int input_size, int output_size)
        : BaseModule(),
          weights({output_size, input_size}, false, "random"),
          bias({output_size, 1}, false, "zeroes") {
    }


Tensor Linear::forward(Tensor& input) {
    Tensor output = weights.matmul(input).add(bias);
    output.grad_ = true;
    output.root_ = gradNode();
    output.root_.module = this;
    output.root_.children.push_back(new gradNode(input.root_));
    last_input = input;
    return output;
}

Tensor MSELoss::forward(Tensor& input, Tensor& correct){
    last_diff = correct.add(input * -1.0);
    float loss = (last_diff^2).sum()/input.size();
    last_diff = last_diff*(-2.0f/input.size());
    Tensor out({1}, true); 
    out[0] = loss;
    out.grad_ = true;
    out.root_ = gradNode();
    out.root_.module = this;
    out.root_.children.push_back(new gradNode(input.root_));
    return out;
}





Tensor Linear::backward(const Tensor& grad, float LR) {
//    std::cout<<"Linear backward"<<std::endl;
//    std::cout<<grad.dims()[0]<<","<<grad.dims()[1]<<" "<<last_input.dims()[0]<<","<<last_input.dims()[1]<<std::endl;
//    std::cout<<grad.matmul(last_input.transpose()).dims()[0]<<","<<grad.matmul(last_input.transpose()).dims()[1]<<std::endl;
    weights = weights.add(grad.matmul(last_input.transpose()) * (-LR));
    bias = bias.add(grad * (-LR));
    return weights.transpose().matmul(grad);
}


Tensor ReLU::forward(Tensor& input){
  
    Tensor output(input.dims(), false, "zeroes");
    Tensor temp(input.dims(),false,"zeroes");
    for(int i = 0; i < input.size(); i++) {
        float x = input.data()[i];
        if (x>0){
          output.data()[i]=x;
          temp.data()[i]=1;
    
        }
    }
    last_diff = temp;
    return output;
}
Tensor ReLU::backward(const Tensor& grad, float LR){

return grad.matmul(last_diff);
};

Tensor MSELoss::backward(const Tensor& grad, float LR) {
    return last_diff;
}

Tensor Sigmoid::forward(Tensor& input) {
    Tensor output(input.dims(), false, "none");
    for(int i = 0; i < input.size(); i++) {
        float x = input.data()[i];
        output.data()[i] = 1.0f / (1.0f + std::exp(-x));
    }
    last_output = output;
    output.grad_ = true;
    output.root_ = gradNode();
    output.root_.module = this;
    output.root_.children.push_back(new gradNode(input.root_));
    return output;
}

Tensor Sigmoid::backward(const Tensor& grad, float LR) {
    Tensor result(grad.dims(), false, "none");
    for (int i = 0; i < grad.size(); i++) {
        result.data()[i] = grad.data()[i] * last_output.data()[i] * (1.0f - last_output.data()[i]);
    }
    return result;
}



void Optim::optimize(Tensor& subject){
        gradNode* current = &subject.root_;

        Tensor accumulator(subject.dims(), false, "none");
        if (subject.dims().size() == 1 && subject.dims()[0] == 1) {
            accumulator[0] = 1.0f; // Start with gradient of 1 for scalar loss
        }

        //std::cout << "Starting optimization..." << std::endl;

        while (current != nullptr) {
            //std::cout << "Current node: " << &(*current) << ", module: " << current->module << std::endl;

            if (current->module == nullptr) {
                //std::cout << "Null module found, skipping..." << std::endl;
                if (current->children.empty()) {
                    //std::cout << "No children, stopping" << std::endl;
                    break;
                }
                current = current->children[0];
                continue;
            }

            accumulator = current->module->backward(accumulator, lr);

            if (current->children.empty()) {
                break;
            }

            if (current->children[0] != nullptr) {
                current = current->children[0];
            } else {
                break;
            }
        }

    }
}
