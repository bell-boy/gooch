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
    const float mu; // momentum coefficnet

    std::vector<Tensor> vel;

    // Constructor: take ownership of the parameter pointers and set lr
    SGD(const std::vector<Tensor>& parameters, float learning_rate, float momentum = 0.9f)
      : params(parameters), lr(learning_rate), mu(momentum) {
        // Initialize momentums
        vel.reserve(params.size());
        for (const auto& p : params) {
            vel.push_back( zeros(p.shape()) );
        }
      }

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
        for (size_t idx = 0; idx < params.size(); ++idx) {
            Tensor& p = params[idx];
            Tensor& v = vel[idx];

            p.TouchGrad();

            size_t N = p.size();
            float* data = p.data().get();
            float* grad = p.grad_data().get();

            float* v_data = v.data().get();

            // momentum update: v = μ * v
            // glas::scal(N, mu, v.data().get());

            // glas::mul_simd(N, mu, v_data);

            glas::mul_cons_simd(N, v_data, mu);

            // momentum update: v += (1 - μ) * grad
            glas::axpy(N, (1.0f - mu), grad, v.data().get());

            // parameter update: θ -= α * v
            glas::axpy(N, -lr, v.data().get(), data);
        }
    }
};

// weights = weights - lr * gradient

} // namespace gooch
