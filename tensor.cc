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
  this->strides_ = std::vector<size_t>(shape.size());
  this->size_ = 1;
  for (int i = shape.size() - 1; i >= 0; i--) {
    this->strides_[i] = this->size_;
    this->size_ *= this->shape_[i];
  }
  this->data_ = std::shared_ptr<float>(new float[this->size_], std::default_delete<float[]>());

}

// View constructor
Tensor::Tensor(std::shared_ptr<float> data, std::vector<size_t> shape, std::vector<size_t> strides, size_t offset) {
  this->data_ = data;
  this->shape_ = shape;
  this->strides_ = strides;
  this->offset_ = offset;
  this->size_ = 1;
  for (int i = shape.size() - 1; i >= 0; i--) {
    this->size_ *= this->shape_[i];
  }
}


Tensor Tensor::operator[](std::vector<Slice> indices) {
  size_t offset = this->offset_;
  std::vector<size_t> new_shape;
  std::vector<size_t> new_strides;
  while (indices.size() < this->shape_.size()) {
    indices.push_back(Slice::all());
  }
  for (size_t i = 0; i < indices.size(); i++) {
    auto it = indices.begin() + i;
    int start = it->start_ < 0 ? it->start_ + this->shape_[i] : it->start_;
    int end = it->end_ < 0 ? it->end_ + this->shape_[i] : it->end_;
    // the new shape is ceil((end - start + 1) / it->step_)
    int size = (end - start + it->step_) / it->step_;
    assert(size > 0);
    if (size > 1) {
      new_shape.push_back(size);
      new_strides.push_back(this->strides_[i] * it->step_);
    }
    offset += start * this->strides_[i];
  }
  return Tensor(this->data_, new_shape, new_strides, offset);
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

}
