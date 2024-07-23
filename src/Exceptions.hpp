#pragma once
#include <exception>
#include "Tensor.hpp"
#include <string_view>

class UnmatchingTensorSizes: public std::exception{
    public:
        UnmatchingTensorSizes(std::string_view _message): message(_message) {}
        std::string message;
        const char* what() const noexcept override{
            return message.c_str();
        }

};
