#include "tensor.h"
// GLAS is a re-implementation of a few kernels from BLAS
// All of the kernels expect the input to be *contiguous* in memory
namespace gooch {
namespace glas {

void axpy(size_t N, float a, const float* x, float* y);
Tensor add(const Tensor& a, const Tensor& b);
void add_(const Tensor& a, const Tensor& b);
Tensor einsum(const Tensor &a, const Tensor &b, const std::string& equation);
void mul_simd(size_t N, const float* x, float* y);
Tensor mul(const Tensor& a, const Tensor& b);
void div_simd(size_t N, const float* x, float* y);
Tensor div(const Tensor& a, const Tensor& b);
void sub_simd(size_t N, const float* x, float* y);
Tensor sub(const Tensor& a, const Tensor& b);
void neg_simd(size_t N, float* y);
Tensor neg(const Tensor& a);
void inv_simd(size_t N, float* y);
Tensor inv(const Tensor& a);
void log_buf(size_t N, float* y);
Tensor log(const Tensor& a);
void exp_buf(size_t N, float* y);
Tensor exp(const Tensor& a);
Tensor einsum(const Tensor& a, const Tensor& b, const std::string& equation);
void root_buf(size_t N, float* y);
Tensor root(const Tensor& a);
void mul_cons_simd(size_t N, float* y, float x);
void inplace_add_square_const(size_t N, float a, const float* x, float* y);
void adam_update(size_t N,
    float* theta,           // parameter buffer
    const float* m,     // first moment
    const float* v,     // second moment
    float lr,
    float eps);
Tensor reduce(const Tensor& a, std::function<float(float, float)> op, std::unordered_set<size_t> axes, float fill);
Tensor reduceSum(const Tensor& a, std::unordered_set<size_t> axes);
Tensor reduceMax(const Tensor& a, std::unordered_set<size_t> axes);
}
}