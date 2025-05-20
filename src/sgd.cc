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
    const float beta; 
    const float epsilon;

    std::vector<Tensor> vel;
    std::vector<Tensor> vel_sq;

    // Constructor: take ownership of the parameter pointers and set lr
    SGD(const std::vector<Tensor>& parameters, float learning_rate, float momentum = 0.9f, float momentum_beta = 0.99f)
      : params(parameters), lr(learning_rate), mu(momentum), beta(momentum_beta), epsilon(1e-8f) {
        // Initialize momentums
        vel.reserve(params.size());
        for (const auto& p : params) {
            vel.push_back( zeros(p.shape()) );
            vel_sq.push_back( zeros(p.shape()) );
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
            Tensor& vsq = vel_sq[idx];

            p.TouchGrad();

            size_t N = p.size();
            float* data = p.data().get();
            float* grad = p.grad_data().get();
            float* v_ptr = v.data().get();
            float* sq_ptr = vsq.data().get();

            // first moment (momentum) update
            glas::mul_cons_simd(N, v_ptr, mu);
            glas::axpy(N, (1.0f - mu), grad, v_ptr);

            // Tensor prod = glas::mul_simd(N, sq_ptr, sq_ptr);

            glas::mul_cons_simd(N, sq_ptr, beta);
            // glas::inplace_add_square_const(N, (1.0f - beta), grad, sq_ptr);
            glas::inplace_add_square_const(N, (1.0f - beta), grad, sq_ptr);



            // // second moment update
            // for (size_t i = 0; i < N; ++i) {
            //     float gval = grad[i];
            //     sq_ptr[i] = beta * sq_ptr[i] + (1.0f - beta) * gval * gval;
            // }

            // parameter update
            // for (size_t i = 0; i < N; ++i) {
            //     data[i] -= lr * (v_ptr[i] / (std::sqrt(sq_ptr[i]) + epsilon));
            // }
            glas::adam_update(N, data, v_ptr, sq_ptr, lr, epsilon);


        }
    }
};

// weights = weights - lr * gradient

} // namespace gooch
