#ifndef LAYER_HPP
#define LAYER_HPP

#include <vector>
#include <functional>
#include <cmath>
#include <stdexcept>

using ActivationFunction = std::function<double(double)>;

namespace Activation {
    inline double relu(double x) { return (x > 0.0) ? x : 0.0; }
    inline double reluDerivative(double x) { return (x > 0.0) ? 1.0 : 0.0; }

    inline double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
    inline double sigmoidDerivative(double x) {
        double s = sigmoid(x);
        return s * (1.0 - s);
    }
}

enum class ActivationType {
  None,
  ReLU,
  Sigmoid,  
};

inline std::pair<ActivationFunction, ActivationFunction> getActivationPair(ActivationType type) {
    using namespace Activation;

    switch (type) {
        case ActivationType::ReLU:
            return { relu, reluDerivative };

        case ActivationType::Sigmoid:
            return { sigmoid, sigmoidDerivative };
        // If it's none or anything else, we choose the default scenario
        default:
            return { ActivationFunction{}, ActivationFunction{} };
    }
}

struct Layer {
    private:
    ActivationFunction activation;
    ActivationFunction activation_derivative;

    public:
    int layer_index;
    int size;
    std::vector<double> z;
    std::vector<double> a;
    std::vector<double> bias;
    std::vector<double> gradient;

    Layer(int index, int size, ActivationType act_type): layer_index(index), size(size), z(size, 0.0), a(size, 0.0) {
        if (size <= 0) {
            throw std::invalid_argument("Layer sizes must be positive");
        }

        if (index != 0) {
            gradient = std::vector<double>(size, 0.0);
            bias = std::vector<double>(size, 0.0);
            activation = getActivationPair(act_type).first;
            activation_derivative = getActivationPair(act_type).second;
        }
    }

    double applyActivation(double x) const {
        if (!activation) {
            throw std::runtime_error("This layer has no activation");
        }

        return activation(x);
    }

    double applyActivationDerivative(double x) const {
        if (!activation_derivative) {
            throw std::runtime_error("This layer has no activation derivative");
        }

        return activation_derivative(x);
    }

    bool hasActivation() const { return static_cast<bool>(activation); }
    bool hasDerivative() const { return static_cast<bool>(activation_derivative); }
};

#endif