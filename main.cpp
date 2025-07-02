#include "tensor.hpp"
#include <iostream>

int main() {
    Tensor a({2, 3});
    Tensor b({3, 2});
    Tensor c = a.matmul(b);

    std::cout << "Matrix multiplication successful! Result dims: ";
    for (int d : c.dims()) {
        std::cout << d << " ";
    }
    std::cout << std::endl;

    return 0;
}
