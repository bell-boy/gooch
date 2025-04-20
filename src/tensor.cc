#include <vector>
#include <memory>
#include <cassert>
#include <numeric>
#include <sstream>
#include <iomanip>
#include "tensor.h"

namespace gooch {

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


View Tensor::operator[](std::vector<Slice> indices) {
  size_t offset = this->offset_;
  std::vector<size_t> new_shape;
  std::vector<int> new_strides;
  assert(indices.size() <= this->shape_.size());
  while (indices.size() < this->shape_.size()) {
    indices.push_back(Slice::all());
  }
  size_t new_size = 1;
  for (size_t i = 0; i < indices.size(); i++) {
    auto it = indices.begin() + i;
    int start = it->start_ < 0 ? it->start_ + this->shape_[i] : it->start_;
    int end = it->end_ < 0 ? it->end_ + this->shape_[i] : it->end_;
    // the new shape is ceil((end - start + 1) / it->step_)
    int dim_size = (end - start + it->step_) / it->step_;
    assert(dim_size > 0);
    if (dim_size > 1) {
      new_shape.push_back(dim_size);
      new_strides.push_back(this->strides_[i] * it->step_);
      new_size *= this->strides_[i] == 0 ? 1 : dim_size;
    }
    offset += start * this->strides_[i];
  }
  return View(this->data_, new_shape, new_strides, offset, new_size);
}

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
    for (size_t i = 0; i < t.shape()[0]; i++) {
      result += recursive_print(t[{i}]);
      if (i < t.shape()[0] - 1) {
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
      for (size_t i = 0; i < a.shape()[0]; i++) {
        recursive_copy(a[{i}], b[{i}]);
      }
    }
  };
  recursive_copy(*this, t);
}

Tensor operator+(const Tensor& a, const Tensor& b) {
  std::vector<size_t> broadcast_shape = Tensor::GetBroadcastShape(a, b);
  Tensor a_broadcast = Tensor::Broadcast(a, broadcast_shape);
  Tensor b_broadcast = Tensor::Broadcast(b, broadcast_shape);
  Tensor result(broadcast_shape);
  std::function<void(Tensor, Tensor, Tensor)> recursive_add = [&recursive_add](Tensor a, Tensor b, Tensor result) {
    if (a.shape().size() == 0) {
      result.data().get()[result.offset()] = a.data().get()[a.offset()] + b.data().get()[b.offset()];
    } else {
      for (size_t i = 0; i < a.shape()[0]; i++) {
        recursive_add(a[{i}], b[{i}], result[{i}]);
      }
    }
  };
  recursive_add(a_broadcast, b_broadcast, result);
  return result;
}

Tensor operator-(const Tensor& a, const Tensor& b) {
  std::vector<size_t> broadcast_shape = Tensor::GetBroadcastShape(a, b);
  Tensor a_broadcast = Tensor::Broadcast(a, broadcast_shape);
  Tensor b_broadcast = Tensor::Broadcast(b, broadcast_shape);
  Tensor result(broadcast_shape);
  std::function<void(Tensor, Tensor, Tensor)> recursive_sub = [&recursive_sub](Tensor a, Tensor b, Tensor result) {
    if (a.shape().size() == 0) {
      result.data().get()[result.offset()] = a.data().get()[a.offset()] - b.data().get()[b.offset()];
    } else {
      for (size_t i = 0; i < a.shape()[0]; i++) {
        recursive_sub(a[{i}], b[{i}], result[{i}]);
      }
    }
  };
  recursive_sub(a_broadcast, b_broadcast, result);
  return result;
}

Tensor operator*(const Tensor& a, const Tensor& b) {
  std::vector<size_t> broadcast_shape = Tensor::GetBroadcastShape(a, b);
  Tensor a_broadcast = Tensor::Broadcast(a, broadcast_shape);
  Tensor b_broadcast = Tensor::Broadcast(b, broadcast_shape);
  Tensor result(broadcast_shape);
  std::function<void(Tensor, Tensor, Tensor)> recursive_mul = [&recursive_mul](Tensor a, Tensor b, Tensor result) {
    if (a.shape().size() == 0) {
      result.data().get()[result.offset()] = a.data().get()[a.offset()] * b.data().get()[b.offset()];
    } else {
      for (size_t i = 0; i < a.shape()[0]; i++) {
        recursive_mul(a[{i}], b[{i}], result[{i}]);
      }
    }
  };
  recursive_mul(a_broadcast, b_broadcast, result);
  return result;
}

Tensor operator/(const Tensor& a, const Tensor& b) {
  std::vector<size_t> broadcast_shape = Tensor::GetBroadcastShape(a, b);
  Tensor a_broadcast = Tensor::Broadcast(a, broadcast_shape);
  Tensor b_broadcast = Tensor::Broadcast(b, broadcast_shape);
  Tensor result(broadcast_shape);
  std::function<void(Tensor, Tensor, Tensor)> recursive_div = [&recursive_div](Tensor a, Tensor b, Tensor result) {
    if (a.shape().size() == 0) {
      result.data().get()[result.offset()] = a.data().get()[a.offset()] / b.data().get()[b.offset()];
    } else {
      for (size_t i = 0; i < a.shape()[0]; i++) {
        recursive_div(a[{i}], b[{i}], result[{i}]);
      }
    }
  };
  recursive_div(a_broadcast, b_broadcast, result);
  return result;
}



}
