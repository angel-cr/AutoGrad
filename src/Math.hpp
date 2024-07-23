#pragma once

#include <cmath>
#include "Tensor.hpp"

namespace MLNet{
    template<SupportedType T>
    Tensor<T> sin(const Tensor<T>& tensor){
        auto result = Tensor<T>{tensor};
        std::vector<T> tensor_values = result.get_value();
        for(auto i=0; i<tensor_values.size(); i++){
            tensor_values[i] = static_cast<T>(std::sin(tensor_values[i]));
        }
        result.set_value(tensor_values);
        result.set_function(TFunction::SIN);
        return result;
    }

    template<SupportedType T>
    Tensor<T> exp(const Tensor<T>& tensor){
        auto result = Tensor<T>{tensor};
        std::vector<T> tensor_values = result.get_value();
        for(auto i=0; i<tensor_values.size(); i++){
            tensor_values[i] = static_cast<T>(std::exp(tensor_values[i]));
        }
        result.set_value(tensor_values);
        result.set_function(TFunction::EXP);
        return result;
    }
}
