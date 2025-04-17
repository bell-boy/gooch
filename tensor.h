#include <vector>
#include <memory>
#include <string>
namespace gooch {

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
//   Tensor t2 = t[{1, Slice::all(), 2}];  // Creates a 1x3x2 tensor
class Tensor {
  std::shared_ptr<float> data_;
  std::vector<size_t> shape_;
  std::vector<size_t> strides_;
  size_t offset_;
  size_t size_;

public:
  Tensor(std::vector<size_t> shape);
  Tensor(std::shared_ptr<float> data, std::vector<size_t> shape, std::vector<size_t> strides, size_t offset);
  Tensor operator[](std::vector<Slice> indices);

  std::shared_ptr<float> data() const;
  std::vector<size_t> shape() const;
  size_t size() const;

  std::string str() const;


};

Tensor zeros(std::vector<size_t> shape);
Tensor ones(std::vector<size_t> shape);

}