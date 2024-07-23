#include<iostream>
#include "Tensor.hpp"
#include "Math.hpp"
#include <vector>

int main(){
    // Test
    auto tensor1 = MLNet::Tensor<float>{{1, 2, 3, 4}};
    auto tensor2 = MLNet::Tensor<float>{{6, 7, 8, 9}};
    std::cout << tensor1 - tensor2 << std::endl;
    auto tensor3 = MLNet::Tensor<float>{{9, 8, 7, 6}};
    std::cout << tensor3 * tensor2 << std::endl;
}
