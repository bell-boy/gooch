#include "tensor.h"
// GLAS is a re-implementation of a few kernels from BLAS
// All of the kernels expect the input to be *contiguous* in memory
namespace gooch {
namespace glas {

void axpy(size_t N, float a, const float* x, float* y);
Tensor add(const Tensor& a, const Tensor& b);
void add_(const Tensor& a, const Tensor& b);
Tensor einsum(const std::string& equation, const Tensor& a, const Tensor& b);

}
}