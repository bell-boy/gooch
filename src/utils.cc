#include "tensor.h"
#include "utils.h"

#include <set>

namespace gooch {
namespace utils {
void BufferCopy(const Tensor& a, float* buffer) {
  std::function<void(Tensor, float*, int, int, int)> recursive_copy = [&](Tensor t, float* buffer, int buffer_offset, int tensor_offset, int N) {
    if (N == 0) {
      buffer[buffer_offset] = t.data().get()[tensor_offset + t.offset()];
    } else {
      for (size_t i = 0; i < t.shape()[N - 1]; i++) {
        size_t buffer_stride = 1;
        for (size_t j = N; j < t.shape().size(); j++) {
          buffer_stride *= t.shape()[j];
        }
        int tensor_stride = t.strides()[N - 1];
        recursive_copy(t, buffer, buffer_offset + i * buffer_stride, tensor_offset + i * tensor_stride, N - 1);
      }
    }
  };
  recursive_copy(a, buffer, 0, 0, a.shape().size());
}

void BufferAssign(const Tensor& a, std::shared_ptr<float> buffer) {
  Tensor b(a.shape(), a.strides(), 0, buffer);
  View a_prime(a);
  a_prime = b;
}

std::shared_ptr<float> broadcast_tensor_to_buf(const Tensor& a, const std::vector<size_t>& shape, size_t size) {
  Tensor broadcast = Tensor::Broadcast(a, shape);
  std::shared_ptr<float> buffer(new float[size], std::default_delete<float[]>());
  utils::BufferCopy(broadcast, buffer.get());
  return buffer;
}

std::vector<int> compute_strides(const std::vector<size_t>& shape) {
  std::vector<int> strides(shape.size());
  int shape_accumulator = 1;
  for (int i = (int) strides.size() - 1; i >= 0; --i) {
    strides[i] = shape_accumulator;
    shape_accumulator *= shape[i];
  }
  return strides;
}
}
}