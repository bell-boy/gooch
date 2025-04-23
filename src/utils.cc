#include "tensor.h"
#include "utils.h"

#include <set>

namespace gooch {
namespace utils {
void BufferCopy(const Tensor& a, float* buffer) {
  std::function<void(Tensor, float*, size_t, size_t, size_t)> recursive_copy = [&](Tensor t, float* buffer, size_t buffer_offset, size_t tensor_offset, size_t N) {
    if (N == 0) {
      buffer[buffer_offset] = t.data().get()[tensor_offset + t.offset()];
    } else {
      for (size_t i = 0; i < t.shape()[N - 1]; i++) {
        size_t buffer_stride = 1;
        for (size_t j = N; j < t.shape().size(); j++) {
          buffer_stride *= t.shape()[j];
        }
        recursive_copy(t, buffer, buffer_offset + i * buffer_stride, tensor_offset + i * t.strides()[N - 1], N - 1);
      }
    }
  };
  recursive_copy(a, buffer, 0, 0, a.shape().size());
}

// Sums 'a' along the axes specified by 'reduced_indices.' Adds into 'buffer.'
void BufferReduce(const Tensor& a, float* buffer, std::set<size_t> reduced_indicies) {
  std::function<void(Tensor, float*, size_t, size_t, size_t)> recursive_reduce = [&](Tensor t, float* buffer, size_t buffer_offset, size_t tensor_offset, size_t N) {
    if (N == 0) {
      buffer[buffer_offset] += t.data().get()[tensor_offset + t.offset()];
    } else {
      for (size_t i = 0; i < t.shape()[N - 1]; i++) {
        size_t new_buffer_offset = reduced_indicies.count(N - 1) ? buffer_offset : buffer_offset + i * t.strides()[N - 1];
        recursive_reduce(t, buffer, new_buffer_offset, tensor_offset + i * t.strides()[N - 1], N - 1);
      }
    }
  };
  recursive_reduce(a, buffer, 0, 0, a.shape().size());
}

void BufferAssign(const Tensor& a, float* const buffer) {
  Tensor b(a.shape(), a.strides(), 0, std::shared_ptr<float>(buffer));
  View a_prime(a);
  a_prime = b;
}
}
}