#pragma once

#include "utils.h"

#include <vector>
#include <memory>
#include <string>
#include <functional>
#include <cassert>
#include <iostream>
#include <numeric>
#include <unordered_set>

namespace gooch {

class Tensor;

namespace detail {
  template<typename T>
  struct is_tensor_type : std::false_type {};

  template<>
  struct is_tensor_type<float> : std::true_type {};

  template<typename T, typename U>
  struct is_tensor_type<std::vector<T, U>> : is_tensor_type<T> {};

  // helper functions for the vector constructor
  template<typename T>
  void get_shape(const T t, std::vector<size_t>& shape) {
    if constexpr (std::is_same<T, float>::value) {
      (void)t;  // Mark parameter as unused
      return;
    } else {
      shape.push_back(t.size());
      get_shape(t[0], shape);
    }
  }

  template<typename T>
  bool check_rectangular(const T t, size_t index, std::vector<size_t>& shape) {
    if constexpr (std::is_same<T, float>::value || std::is_same<T, std::vector<float>>::value) {
      return true;
    } else {
      bool result = true;
      for (size_t i = 0; i < t.size(); i++) {
        if (shape[index] != t[i].size()) {
          return false;
        }
        result = result && check_rectangular(t[i], index + 1, shape);
      }
      return result;
    }
  }

  template<typename T>
  void recursive_fill(const T t, std::shared_ptr<float> data, size_t offset, size_t index, std::vector<int>& strides) {
    if constexpr (std::is_same<T, float>::value) {
      data.get()[offset] = t;
    } else {
      for (size_t i = 0; i < t.size(); i++) {
        recursive_fill(t[i], data, offset + i * strides[index], index + 1, strides);
      }
    }
  }
}


// A class representing a slice of a tensor.
// Used when indexing a tensor.
// Slices are *inclusive* of the start and end indices.
class Slice{
public:
  int start_;
  int end_;
  int step_;

  Slice(int start, int end, int step);
  Slice(int start, int end);
  Slice(int index);

  static Slice all();
};

class View;

// A class representing a multi-dimensional tensor.
// This class provides functionality for creating and manipulating tensors with arbitrary dimensions.
// The tensor is stored in contiguous memory and supports efficient indexing operations.
// 
// Example usage:
//   Tensor t({2, 3, 4});  // Creates a 2x3x4 tensor
//   Tensor t2 = t[{1, Slice::all(), 2}];  // Creates a 3x2 tensor
class Tensor {
protected:
  std::vector<size_t> shape_;
  std::vector<int> strides_;


  std::shared_ptr<float> data_;
  std::shared_ptr<std::shared_ptr<float>> grad_;

  size_t offset_;
  size_t size_;
  size_t original_size_; // the size of the tensor at initialization, use to properly size the grad buffer

public:
  std::function<void(Tensor)> grad_fn_;
  bool is_leaf_;
  Tensor(std::vector<size_t> shape); // creates a tensor with no data
  Tensor(std::vector<size_t> shape, std::vector<int> strides, size_t offset, Tensor t); // creates a view of t
  Tensor(std::vector<size_t> shape, std::vector<int> strides, size_t offset, std::shared_ptr<float> data); // create a new tensor with the given shape and strides, and data

  template<typename... Args>
  View operator()(Args... indices) const;
  friend std::ostream& operator<<(std::ostream& os, const Tensor& t);

  std::shared_ptr<float> data() const;
  std::shared_ptr<float> grad_data() const;
  void TouchGrad() const;
  std::vector<size_t> shape() const;
  size_t size() const;
  size_t offset() const;
  std::vector<int> strides() const;
  std::string str() const;

  Tensor grad() const;
  void Backward();
  void ZeroGrad();



  static std::vector<size_t> GetBroadcastShape(const Tensor& a, const Tensor& b);
  static Tensor Broadcast(const Tensor& a, const std::vector<size_t>& shape);

};

class View : public Tensor {
public:
  View(std::vector<size_t> shape, std::vector<int> strides, size_t offset, Tensor t);
  View(const Tensor& t);
  void operator=(const Tensor& other);
};


Tensor zeros(std::vector<size_t> shape);
Tensor ones(std::vector<size_t> shape);
Tensor randn(std::vector<size_t> shape);
void propagate_grad(const Tensor& grad, const Tensor& op);


template<typename... Args>
View Tensor::operator()(Args... indices) const {
  std::vector<Slice> slices = {indices...};
  std::vector<size_t> new_shape;
  std::vector<int> new_strides;
  size_t new_size = 1;
  size_t new_offset = offset_;
  // using given slices
  for (size_t i = 0; i < slices.size(); i++) {
    int start = slices[i].start_ < 0 ? slices[i].start_ + shape_[i] : slices[i].start_;
    int end = slices[i].end_ < 0 ? slices[i].end_ + shape_[i] : slices[i].end_;
    int dim_size = (end - start + slices[i].step_) / slices[i].step_;
    assert(dim_size > 0);
    if (dim_size > 1) {
      new_shape.push_back(dim_size);
      new_strides.push_back(strides_[i] * slices[i].step_);
      new_size *= dim_size;
    }
    new_offset += start * strides_[i];
  }
  // adding any missing slices
  for (size_t i = slices.size(); i < shape_.size(); i++) {
    new_shape.push_back(shape_[i]);
    new_strides.push_back(strides_[i]);
    new_size *= shape_[i];
  }
  View result = View(new_shape, new_strides, new_offset, *this);
  Tensor this_tensor = *this;
  result.grad_fn_ = [this_tensor, new_shape , slices](Tensor grad) {
    std::vector<int> new_grad_strides = utils::compute_strides(new_shape);
    size_t new_grad_offset = 0;
    for (size_t i = 0; i < slices.size(); i++) {
      int start = slices[i].start_ < 0 ? slices[i].start_ + this_tensor.shape()[i] : slices[i].start_;
      new_grad_offset += start * utils::compute_strides(this_tensor.shape())[i];
    }
    Tensor new_grad = zeros(this_tensor.shape());
    View(new_shape, new_grad_strides, new_grad_offset, new_grad) = grad;
    propagate_grad(new_grad, this_tensor);
  };
  return result;
}


// Vector constructor
template<typename T>
Tensor FromVector(T data) {
  static_assert(detail::is_tensor_type<T>::value, "Data must be a single float or a vector of vectors of ... of floats");
  if constexpr (std::is_same<T, float>::value) {
    Tensor t(std::vector<size_t>{});
    *t.data() = data;
    return t;
  } else {
    // 1. get the shape
    std::vector<size_t> shape;
    detail::get_shape(data, shape);
    // 2. check that the tensor is rectangular
    assert(detail::check_rectangular(data, 1, shape));
    // 3. create the data buffer and copy the data
    std::vector<int> strides(shape.size());
    size_t size = 1;
    for (int i = shape.size() - 1; i >= 0; i--) {
      strides[i] = size;
      size *= shape[i];
    }
    std::shared_ptr<float> data_ptr = std::shared_ptr<float>(new float[size], std::default_delete<float[]>());
    detail::recursive_fill(data, data_ptr, 0, 0, strides);
    Tensor t(shape, strides, 0, data_ptr);
    return t;
  }
}

Tensor operator+(const Tensor& a, const Tensor& b);
Tensor operator-(const Tensor& a, const Tensor& b);
Tensor operator*(const Tensor& a, const Tensor& b);
Tensor operator/(const Tensor& a, const Tensor& b);
Tensor operator-(const Tensor& a);
Tensor reshape(const Tensor& a);
Tensor Einsum(const Tensor& a, const Tensor& b, const std::string& equation);
Tensor reduceSum(const Tensor& a, std::unordered_set<size_t> axes);
Tensor logSumExp(const Tensor& a, std::unordered_set<size_t> axes);
Tensor crossEntropyLoss(const Tensor& a, std::vector<size_t> correct);
}