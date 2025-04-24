#pragma once

#include <set>
#include <vector>
#include <memory>

namespace gooch {
class Tensor;
namespace utils {

void BufferCopy(const Tensor& a, float* buffer);
void BufferAssign(const Tensor& a, std::shared_ptr<float> buffer);
std::shared_ptr<float> broadcast_tensor_to_buf(const Tensor& a, const std::vector<size_t>& shape, size_t size);
std::vector<int> compute_strides(const std::vector<size_t>& shape);
}
}