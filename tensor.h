#include <vector>
#include <memory>
#include <string>
#include <functional>
#include <cassert>
#include <iostream>
namespace gooch {

template<typename T>
struct is_tensor_type : std::false_type {};

template<>
struct is_tensor_type<float> : std::true_type {};

template<typename T, typename U>
struct is_tensor_type<std::vector<T, U>> : is_tensor_type<T> {};

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

// A class representing a multi-dimensional tensor.
// This class provides functionality for creating and manipulating tensors with arbitrary dimensions.
// The tensor is stored in contiguous memory and supports efficient indexing operations.
// 
// Example usage:
//   Tensor t({2, 3, 4});  // Creates a 2x3x4 tensor
//   Tensor t2 = t[{1, Slice::all(), 2}];  // Creates a 3x2 tensor
class Tensor {
  std::shared_ptr<float> data_;
  std::vector<size_t> shape_;
  std::vector<size_t> strides_;
  std::function<void(Tensor)> grad_fn_;
  size_t offset_;
  size_t size_;

public:
  Tensor(std::vector<size_t> shape);
  Tensor(std::shared_ptr<float> data, std::vector<size_t> shape, std::vector<size_t> strides, size_t offset);
  Tensor operator[](std::vector<Slice> indices);

  std::shared_ptr<float> data() const;
  std::vector<size_t> shape() const;
  size_t size() const;
  size_t offset() const;

  std::string str() const;

};

Tensor zeros(std::vector<size_t> shape);
Tensor ones(std::vector<size_t> shape);

// helper functions for the vector constructor
template<typename T>
void get_shape(const T t, std::vector<size_t>& shape) {
  if constexpr (std::is_same<T, float>::value) {
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
void recursive_fill(const T t, std::shared_ptr<float> data, size_t offset, size_t index, std::vector<size_t>& strides) {
  if constexpr (std::is_same<T, float>::value) {
    data.get()[offset] = t;
  } else {
    for (size_t i = 0; i < t.size(); i++) {
      recursive_fill(t[i], data, offset + i * strides[index], index + 1, strides);
    }
  }
}

// Vector constructor
template<typename T>
Tensor FromVector(T data) {
  static_assert(is_tensor_type<T>::value, "Data must be a single float or a vector of vectors of ... of floats");
  if constexpr (std::is_same<T, float>::value) {
    Tensor t(std::vector<size_t>{});
    *t.data() = data;
    return t;
  } else {
    // 1. get the shape
    std::vector<size_t> shape;
    get_shape(data, shape);
    // 2. check that the tensor is rectangular
    assert(check_rectangular(data, 1, shape));
    // 3. create the data buffer and copy the data
    std::vector<size_t> strides(shape.size());
    size_t size = 1;
    for (int i = shape.size() - 1; i >= 0; i--) {
      strides[i] = size;
      size *= shape[i];
    }
    std::shared_ptr<float> data_ptr = std::shared_ptr<float>(new float[size], std::default_delete<float[]>());
    recursive_fill(data, data_ptr, 0, 0, strides);
    Tensor t(data_ptr, shape, strides, 0);
    return t;
  }
}
}