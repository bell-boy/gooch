#include "tensor.h"

#include <set>

namespace gooch {
namespace utils {
void BufferCopy(const Tensor& a, float* buffer);
void BufferReduce(const Tensor& a, float* buffer, std::set<size_t> reduced_indicies);
void BufferAssign(const Tensor& a, float* const buffer);
}
}