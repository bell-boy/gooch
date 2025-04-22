#include "tensor.h"
#include "glas.h"

#include <vector>
#include <memory>
#include <cassert>
#include <numeric>
#include <sstream>
#include <iomanip>
#include <immintrin.h>
#include <set>

namespace gooch {
namespace detail {
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
}

// Standard tensor constructor with uninitialized data
Tensor::Tensor(std::vector<size_t> shape) {
  this->shape_ = shape;
  this->strides_ = std::vector<int>(shape.size());
  this->size_ = 1;
  this->offset_ = 0;
  for (int i = shape.size() - 1; i >= 0; i--) {
    this->strides_[i] = this->size_;
    this->size_ *= this->shape_[i];
  }
  this->data_ = std::shared_ptr<float>(new float[this->size_], std::default_delete<float[]>());

}

// View constructor
Tensor::Tensor(std::shared_ptr<float> data, std::vector<size_t> shape, std::vector<int> strides, size_t offset, size_t size) : data_(data), shape_(shape), strides_(strides), offset_(offset), size_(size) {}


std::ostream& operator<<(std::ostream& os, const Tensor& t) {
  os << t.str();
  return os;
}

std::shared_ptr<float> Tensor::data() const {
  return this->data_;
}

std::vector<size_t> Tensor::shape() const {
  return this->shape_;
}

size_t Tensor::size() const {
  return this->size_;
}

size_t Tensor::offset() const {
  return this->offset_;
}

std::vector<int> Tensor::strides() const {
  return this->strides_;
}

Tensor zeros(std::vector<size_t> shape) {
  Tensor t(shape);
  std::fill(t.data().get(), t.data().get() + t.size(), 0.0f);
  return t;
}

Tensor ones(std::vector<size_t> shape) {
  Tensor t(shape);
  std::fill(t.data().get(), t.data().get() + t.size(), 1.0f);
  return t;
}


std::string Tensor::str() const {
  std::stringstream ss;
  ss << "Tensor of shape (";
  for (size_t i = 0; i < this->shape_.size(); i++) {
    ss << this->shape_[i];
    if (i < this->shape_.size() - 1) {
      ss << ", ";
    }
  }
  ss << ")" << std::endl;

  std::function<std::string(Tensor)> recursive_print = [&](Tensor t) -> std::string {
    if (t.shape().size() == 0) {
      std::string result = "";
      std::stringstream temp;
      temp << std::fixed << std::setprecision(3) << t.data().get()[t.offset()];
      result += temp.str();
      return result;
    }
    std::string result = "[";
    for (int i = 0; i < (int) t.shape()[0]; i++) {
      result += recursive_print(t(i));
      if (i < (int) t.shape()[0] - 1) {
        result += t.shape().size() == 1 ? ", " : ",\n";
      }
    }
    result += "]";
    return result;
  };
  return ss.str() + recursive_print(*this);
}

Slice::Slice(int start, int end, int step) {
  this->start_ = start;
  this->end_ = end;
  this->step_ = step;
}

Slice::Slice(int start, int end) {
  this->start_ = start;
  this->end_ = end;
  this->step_ = 1;
}

Slice::Slice(int index) {
  this->start_ = index;
  this->end_ = index;
  this->step_ = 1;
}

Slice Slice::all() {
  return Slice(0, -1, 1);
}

View::View(std::shared_ptr<float> data, std::vector<size_t> shape, std::vector<int> strides, size_t offset, size_t size) : Tensor(data, shape, strides, offset, size) {}

std::vector<size_t> Tensor::GetBroadcastShape(const Tensor& a, const Tensor& b) {
  Tensor larger = a.shape().size() > b.shape().size() ? a : b;
  Tensor smaller = a.shape().size() > b.shape().size() ? b : a;
  // pad the front of the smaller tensor shape with 1s
  std::vector<size_t> padded_shape(larger.shape().size() - smaller.shape().size(), 1);
  for (size_t i = 0; i < smaller.shape().size(); i++) {
    padded_shape.push_back(smaller.shape()[i]);
  }
  std::vector<size_t> resulting_shape;
  for (size_t i = 0; i < larger.shape().size(); i++) {
    // assert that the shapes are compatible
    if (larger.shape()[i] != padded_shape[i] && larger.shape()[i] != 1 && padded_shape[i] != 1) {
      throw std::invalid_argument("Shapes are not broadcastable");
    }
    resulting_shape.push_back(std::max(larger.shape()[i], padded_shape[i]));
  }
  return resulting_shape;
}

Tensor Tensor::Broadcast(const Tensor& a, const std::vector<size_t>& shape) {
  std::vector<int> new_strides(shape.size());
  for (size_t i = 0; i < shape.size(); i++) {
    // a might be smaller than shape, so this is like padding with 1s
    size_t a_index = i - (shape.size() - a.shape().size());
    if (i < (shape.size() - a.shape().size())) {
      new_strides[i] = 0;
    } else {
      assert(a.shape()[a_index] == shape[i] || a.shape()[a_index] == 1);
      new_strides[i] = a.shape()[a_index] == shape[i] ? a.strides()[a_index] : 0;
    }
  }
  return Tensor(a.data(), shape, new_strides, a.offset(), a.size());
}

void View::operator=(const Tensor& other) {
  Tensor t = Tensor::Broadcast(other, this->shape_);
  std::function<void(Tensor, Tensor)> recursive_copy = [&recursive_copy](Tensor a, Tensor b) {
    if (a.shape().size() == 0) {
      a.data().get()[a.offset()] = b.data().get()[b.offset()];
    } else {
      for (int i = 0; i < (int) a.shape()[0]; i++) {
        recursive_copy(a(i), b(i));
      }
    }
  };
  recursive_copy(*this, t);
}

Tensor operator+(const Tensor& a, const Tensor& b) {
  std::vector<size_t> broadcast_shape = Tensor::GetBroadcastShape(a, b);
  size_t size = std::accumulate(broadcast_shape.begin(), broadcast_shape.end(), 1, std::multiplies<size_t>());
  Tensor broadcast_a = Tensor::Broadcast(a, broadcast_shape);
  Tensor broadcast_b = Tensor::Broadcast(b, broadcast_shape);

  float* a_buffer = new float[size];
  std::shared_ptr<float> b_buffer(new float[size], std::default_delete<float[]>());
  detail::BufferCopy(broadcast_a, a_buffer);
  detail::BufferCopy(broadcast_b, b_buffer.get());

  glas::axpy(size, 1.0f, a_buffer, b_buffer.get());

  std::vector<int> strides(broadcast_shape.size());
  for (int i = (int) broadcast_shape.size() - 1, j = 1; i >= 0; i--) {
    strides[i] = j;
    j *= broadcast_shape[i];
  }
  Tensor result(b_buffer, broadcast_shape, strides, 0, size);
  result.is_leaf_ = false;
  result.grad_fn_ = [broadcast_a, broadcast_b](Tensor grad) {
    (void) grad;
  };
  return result;
}

}