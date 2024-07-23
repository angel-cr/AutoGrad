/*
    Tensor class that mimics the torch.Tensor one with autograd.
    Start with some basic operations and add on more afterwards
    Example Usage:
    auto x = Tensor([1.]);
    auto y = Tensor([2.]);
    auto z = x + y // returns [3.]
*/
#pragma once

#include <cstdint>
#include <concepts>
#include <vector>
#include <iostream>
#include <string>
#include <limits>
#include "Exceptions.hpp"

template <typename T>
concept SupportedType = std::is_floating_point<T>::value;

enum class TFunction{
    ADDITION,
    SUBSTRACTION,
    PRODUCT,
    DIVISION,
    SIN,
    EXP,
    NO_OPERATION // For Tensors that are not a result of any operation between tensor (i.e. newly created Tensors)
};

namespace MLNet
{
    template <SupportedType T>
    class Tensor{
        public:
            Tensor();
            Tensor(const std::vector<T>& _value);
            Tensor(const T _value);
            Tensor(const Tensor& other);
            Tensor(Tensor&& other);
            Tensor(std::vector<T>&& _value);
            Tensor& operator=(const Tensor& other);
            Tensor& operator=(Tensor&& other);
            ~Tensor() = default;

            friend std::ostream& operator<< <>(std::ostream& stream, const Tensor& tensor);

            // basic arithmetic operations: +,-,/,*
            Tensor operator+(const Tensor<T>& other);
            Tensor operator-(const Tensor<T>& other);
            Tensor operator*(const Tensor<T>& other);
            Tensor operator/(const Tensor<T>& other);

            // getters / setters
            void set_value(const std::vector<T>& _value);
            std::vector<T> get_value() const;
            void set_function(TFunction _function);
            TFunction get_function() const;

        private:
            std::vector<T> value; // The class will handle only a 1-dimensional vectors for now
            TFunction function = TFunction::NO_OPERATION;

    };

    template <SupportedType T>
    Tensor<T>::Tensor() {}

    template <SupportedType T>
    Tensor<T>::Tensor(const std::vector<T>& _value): value(_value) {}

    template <SupportedType T>
    Tensor<T>::Tensor(const T _value): value(std::vector<T>{_value}) {}

    template <SupportedType T>
    Tensor<T>::Tensor(const Tensor& other): value(other.value) {}

    template <SupportedType T>
    Tensor<T>::Tensor(std::vector<T>&& _value): value(std::move(_value)) {}

    template <SupportedType T>
    Tensor<T>::Tensor(Tensor<T>&& other): value(std::move(other.value)) {}

    template <SupportedType T>
    Tensor<T>& Tensor<T>::operator=(const Tensor& other){
        if(this != &other)
            this.value = other.value;
        return *this;
    }

    template <SupportedType T>
    Tensor<T>& Tensor<T>::operator=(Tensor&& other){
        if(this != &other){
            this.value = std::move(other.value);
            other.value.clear();
        }
        return *this;
    }

    template <SupportedType T>
    std::ostream& operator<<(std::ostream& stream, const Tensor<T>& tensor){
        stream << "[";
        for(auto& val: tensor.value)
            stream << val << " ";
        stream << "]";
        return stream;
    }


    // ----------------------------- Getters and Setters -----------------------------------
    template <SupportedType T>
    std::vector<T> Tensor<T>::get_value() const{
        return value;
    }

    template <SupportedType T>
    void Tensor<T>::set_value(const std::vector<T>& _value){
        this->value = _value;
    }

    template <SupportedType T>
    TFunction Tensor<T>::get_function() const{
        return this->function;
    }

    template <SupportedType T>
    void Tensor<T>::set_function(TFunction _function){
        this->function = _function;
    }

    // ----------------------------- Arithmetic operations ----------------------------------
    template <SupportedType T>
    Tensor<T> Tensor<T>::operator+(const Tensor<T>& other){
        // Check if both tensors have the same shape
        if(this->value.size() != other.value.size()){
            std::string message = "Tensor a's shape (" + std::to_string(this->value.size()) + ") is different than Tensor b's shape (" + std::to_string(other.value.size()) + "). Both Tensors should have the same shape.";
            throw UnmatchingTensorSizes(message);
        }
        Tensor<T> result;
        std::size_t tensor_len = this->value.size();

        for(auto i=0; i<tensor_len; i++){
            result.value.push_back(this->value[i] + other.value[i]);
        }
        result.function = TFunction::ADDITION;
        return result;
    }

    template <SupportedType T>
    Tensor<T> Tensor<T>::operator-(const Tensor<T>& other){
        // Check if both tensors have the same shape
        if(this->value.size() != other.value.size()){
            std::string message = "Tensor a's shape (" + std::to_string(this->value.size()) + ") is different than Tensor b's shape (" + std::to_string(other.value.size()) + "). Both Tensors should have the same shape.";
            throw UnmatchingTensorSizes(message);
        }
        Tensor<T> result;
        std::size_t tensor_len = this->value.size();

        for(auto i=0; i<tensor_len; i++){
            result.value.push_back(this->value[i] - other.value[i]);
        }
        result.function = TFunction::SUBSTRACTION;
        return result;
    }

    template <SupportedType T>
    Tensor<T> Tensor<T>::operator*(const Tensor<T>& other){
        // Check if both tensors have the same shape
        if(this->value.size() != other.value.size()){
            std::string message = "Tensor a's shape (" + std::to_string(this->value.size()) + ") is different than Tensor b's shape (" + std::to_string(other.value.size()) + "). Both Tensors should have the same shape.";
            throw UnmatchingTensorSizes(message);
        }
        Tensor<T> result;
        std::size_t tensor_len = this->value.size();

        for(auto i=0; i<tensor_len; i++){
            result.value.push_back(this->value[i] * other.value[i]);
        }
        result.function = TFunction::PRODUCT;
        return result;
    }

    template <SupportedType T>
    Tensor<T> Tensor<T>::operator/(const Tensor<T>& other){
        // Check if both tensors have the same shape
        if(this->value.size() != other.value.size()){
            std::string message = "Tensor a's shape (" + std::to_string(this->value.size()) + ") is different than Tensor b's shape (" + std::to_string(other.value.size()) + "). Both Tensors should have the same shape.";
            throw UnmatchingTensorSizes(message);
        }
        Tensor<T> result;
        std::size_t tensor_len = this->value.size();

        for(auto i=0; i<tensor_len; i++){
            if(other.value[i] == 0){
                result.value.push_back(std::numeric_limits<T>::min()); // assign minimum value of type T (mimic NaN)
            } else{
                result.value.push_back(this->value[i] / other.value[i]);
            }
        }
        result.function = TFunction::DIVISION;
        return result;
    }
}
