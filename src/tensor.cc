#include "tensor.h"
#include "glas.h"
#include "utils.h"

#include <vector>
#include <memory>
#include <cassert>
#include <numeric>
#include <sstream>
#include <iomanip>
#include <immintrin.h>
#include <set>

namespace gooch {

// Standard tensor constructor with uninitialized data
Tensor::Tensor(std::vector<size_t> shape) {
  shape_ = shape;
  strides_ = std::vector<int>(shape.size());
  size_ = 1;
  offset_ = 0;
  for (int i = shape.size() - 1; i >= 0; i--) {
    strides_[i] = size_;
    size_ *= shape_[i];
  }
  data_ = std::shared_ptr<float>(new float[size_], std::default_delete<float[]>());
  grad_ = std::shared_ptr<std::shared_ptr<float>>(new std::shared_ptr<float>(nullptr));
  original_size_ = size_;
}

// View constructor
Tensor::Tensor(std::vector<size_t> shape, std::vector<int> strides, size_t offset, Tensor t) : shape_(shape), strides_(strides), data_(t.data_), grad_(t.grad_), offset_(offset), size_(t.size_), original_size_(t.original_size_) {}

std::ostream& operator<<(std::ostream& os, const Tensor& t) {
  os << t.str();
  return os;
}

// new tensor w/ data
Tensor::Tensor(std::vector<size_t> shape, std::vector<int> strides, size_t offset, std::shared_ptr<float> data) : shape_(shape), strides_(strides), data_(data), grad_(std::shared_ptr<std::shared_ptr<float>>(new std::shared_ptr<float>(nullptr))), offset_(offset), size_(std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>())), original_size_(std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>())) {}

std::shared_ptr<float> Tensor::data() const {
  return this->data_;
}

std::shared_ptr<float> Tensor::grad_data() const {
  return *this->grad_;
}

void Tensor::TouchGrad() const {
  if (*this->grad_ == nullptr) {
    *this->grad_ = std::shared_ptr<float>(new float[original_size_], std::default_delete<float[]>());
    std::fill((*this->grad_).get(), (*this->grad_).get() + original_size_, 0.0f);
  }
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

Tensor Tensor::grad() const {
  if (*grad_ == nullptr) {
    //throw std::runtime_error("Gradient not set");
    *grad_ = std::shared_ptr<float>(new float[size_], std::default_delete<float[]>());
    std::fill((*grad_).get(), (*grad_).get() + size_, 0.0f);
  }
  Tensor t(shape_, strides_, offset_, *grad_);
  t.size_ = size_;
  t.original_size_ = original_size_;
  return t;
}

// backward is only defined on scalar tensors
void Tensor::Backward() {
  if (size_ != 1) {
    std::invalid_argument("Backward can only be called on scalar tensors");
  }
  if (!grad_fn_) {
    std::invalid_argument("Tensor must have grad function defined");
  }
  grad_fn_(FromVector(1.0f));
}

void Tensor::ZeroGrad() {
  grad_ = nullptr;
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

View::View(std::vector<size_t> shape, std::vector<int> strides, size_t offset, Tensor t) : Tensor(shape, strides, offset, t) {}

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
      new_strides[i] = a.strides()[a_index];
    }
  }
  return Tensor(shape, new_strides, a.offset(), a);
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

View::View(const Tensor& t) : Tensor(t.shape(), t.strides(), t.offset(), t.data()) {}

void update_grad(const Tensor& grad, const Tensor& op) {
  std::set<size_t> reduced_indices;

  Tensor broadcast = Tensor::Broadcast(op, grad.shape());

  for (size_t i = 0; i < broadcast.shape().size(); ++i) {
    if (broadcast.strides()[i] == 0) {
      reduced_indices.insert(i);
    }
  }

  // TODO: investigate optimization for buffer-reduce -- the only reduced indicies will be on the inside
  std::shared_ptr<float> buffer(new float[op.size()], std::default_delete<float[]>());
  std::fill(buffer.get(), buffer.get() + op.size(), 0.0f);
  utils::BufferReduce(grad, buffer.get(), reduced_indices);

  op.TouchGrad();

  Tensor buf_tensor(op.shape(), op.strides(), 0, buffer);

  if (op.grad_fn_) op.grad_fn_(grad);

  glas::add_(grad, op.grad());
}

Tensor operator+(const Tensor& a, const Tensor& b) {
  Tensor result = glas::add(a, b);
  result.grad_fn_ = [a, b] (Tensor grad) {
    update_grad(grad, a);
    update_grad(grad, b);
  };
  return result;
}

Tensor operator*(const Tensor& a, const Tensor& b) {
  Tensor result = glas::mul(a, b);
  result.grad_fn_ = [a, b] (Tensor grad) {
    Tensor a_grad = glas::mul(grad, b);
    Tensor b_grad = glas::mul(grad, a);
    update_grad(a_grad, a);
    update_grad(b_grad, b);
  };
  return result;
}

Tensor Reduce(const Tensor& a, const std::set<size_t>& reduced_indicies) {
  std::shared_ptr<float> buffer(new float[a.size()], std::default_delete<float[]>());
  std::fill(buffer.get(), buffer.get() + a.size(), 0.0f);
  utils::BufferReduce(a, buffer.get(), reduced_indicies);
  Tensor result = Tensor(a.shape(), a.strides(), 0, buffer);
  result.grad_fn_ = [a, reduced_indicies](Tensor grad) {
    a.TouchGrad();
    // need to "unreduce" the gradient
    std::vector<int> new_strides(a.shape().size());
    for (size_t i = 0; i < a.shape().size(); i++) {
      if (reduced_indicies.count(i)) {
        new_strides[i] = 0;
      } else {
        new_strides[i] = a.strides()[i];
      }
    }
    Tensor unreduced = Tensor(a.shape(), new_strides, grad.offset(), grad.data()); 
    glas::add_(unreduced, a.grad());
  };
  return result;
}

Tensor Einsum(const Tensor& a, const Tensor& b, const std::string& equation) {
  Tensor result = glas::einsum(a, b, equation); 
  result.grad_fn_ = [a, b, equation](Tensor grad) {
    // decompose equation into a, b and c
    std::string a_string; 
    std::string b_string; 
    std::string c_string; 

    for(size_t i = 0, state = 0; i < equation.size(); i++) {
      switch (state)
      {
      case 0:
        if (equation[i] == ',') state++;
        else a_string.push_back(equation[i]);
        break;
      case 1:
        if (equation[i] == '-' && equation[i + 1] == '>') {
          i += 2;
          state++;
        } else b_string.push_back(equation[i]);
        break;
      case 2:
        c_string.push_back(equation[i]);
        break; 
      default:
        break;
      }
    }


    Tensor a_grad = glas::einsum(grad, b, c_string + ", " + b_string + " -> " + a_string);
    std::cout << a_grad << std::endl;
    Tensor b_grad = glas::einsum(grad, a, c_string + ", " + a_string + " -> " + b_string);

    std::cout  << b_grad << std::endl;


    update_grad(a.grad(), a_grad);
    update_grad(b.grad(), b_grad);
  };
  return result;
}
}