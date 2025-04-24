// sgd_optimizer.cc

#include "tensor.h"
#include "glas.h"
#include <vector>
#include <algorithm>
#include <cstddef>

namespace gooch {

struct SGD {
    std::vector<Tensor> params;
    float lr; // learning rate

    // Constructor: take ownership of the parameter pointers and set lr
    SGD(const std::vector<Tensor>& parameters, float learning_rate)
      : params(parameters), lr(learning_rate) {}

    // // Clear gradients on all parameters
    // void zero_grad() {
    //     for (auto& p : params) {
    //         p.TouchGrad();
    //         float* g = p.grad_data().get() + p.offset();
    //         std::fill(g, g + p.size(), 0.0f);
    //     }
    // }

    // Perform one SGD update: θ <- θ - lr * grad
    void step() {
        for (auto& p : params) {

            glas::axpy(p.size(), -lr, p.grad_data().get(), p.data().get());
        }
    }
};

// weights = weights - lr * gradient

} // namespace gooch
