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
Tensor einsum(const std::string& equation, const Tensor& a, const Tensor& b);

}
}