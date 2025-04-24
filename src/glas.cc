#include "glas.h"
#include "tensor.h"
#include "utils.h"

#include <immintrin.h>
#include <map>
#include <cmath>
#include <unordered_set>

namespace gooch {
namespace glas {

void adam_update(size_t N,
    float* theta,           // parameter buffer
    const float* m,     // first moment
    const float* v,     // second moment
    float lr,
    float eps) {
const __m256 lr_vec = _mm256_set1_ps(lr);
const __m256 eps_vec = _mm256_set1_ps(eps);
for (size_t i = 0; i < N - N%8; i += 8) {
__m256 m_vec = _mm256_loadu_ps(m + i);
__m256 v_vec = _mm256_loadu_ps(v + i);
v_vec = _mm256_sqrt_ps(v_vec);
v_vec = _mm256_add_ps(v_vec, eps_vec);
__m256 upd = _mm256_div_ps(m_vec, v_vec);
upd = _mm256_mul_ps(lr_vec, upd);
__m256 theta_vec = _mm256_loadu_ps(theta + i);
theta_vec = _mm256_sub_ps(theta_vec, upd);
_mm256_storeu_ps(theta + i, theta_vec);
}
for (size_t i = N - N%8; i < N; ++i) {
theta[i] -= lr * (m[i] / (std::sqrt(v[i]) + eps));
}
}


void inplace_add_square_const(size_t N, float a, const float* x, float* y) {
  const __m256 alpha_vec = _mm256_set1_ps(a);
  for (size_t i = 0; i < (N - N % 8); i += 8) {
    __m256 x_vec = _mm256_loadu_ps(x + i); 
    x_vec = _mm256_mul_ps(x_vec, x_vec);
    x_vec = _mm256_mul_ps(alpha_vec, x_vec);
    __m256 y_vec = _mm256_loadu_ps(y + i);
    y_vec = _mm256_add_ps(x_vec, y_vec);
    _mm256_storeu_ps(y + i, y_vec);
  }
  for (size_t i = (N - N % 8); i < N; i++) {
    y[i] += x[i] * x[i] * a;
  }
}



void axpy(size_t N, float a, const float* x, float* y) {
  const __m256 alpha_vec = _mm256_set1_ps(a);
  for (size_t i = 0; i < (N - N % 8); i += 8) {
    __m256 x_vec = _mm256_loadu_ps(x + i); 
    x_vec = _mm256_mul_ps(alpha_vec, x_vec);
    __m256 y_vec = _mm256_loadu_ps(y + i);
    y_vec = _mm256_add_ps(x_vec, y_vec);
    _mm256_storeu_ps(y + i, y_vec);
  }
  for (size_t i = (N - N % 8); i < N; i++) {
    y[i] += a * x[i];
  }
}

Tensor add(const Tensor& a, const Tensor& b) {
  std::vector<size_t> broadcast_shape = Tensor::GetBroadcastShape(a, b);
  size_t broadcast_size = std::accumulate(broadcast_shape.begin(), broadcast_shape.end(), 1, std::multiplies<size_t>());

  std::shared_ptr<float> a_buffer = utils::broadcast_tensor_to_buf(a, broadcast_shape, broadcast_size);
  std::shared_ptr<float> b_buffer = utils::broadcast_tensor_to_buf(b, broadcast_shape, broadcast_size);

  glas::axpy(broadcast_size, 1.0f, a_buffer.get(), b_buffer.get());

  return Tensor(broadcast_shape, utils::compute_strides(broadcast_shape), 0, b_buffer);
}

// in-place add, b += a
void add_(const Tensor& a, const Tensor& b) {
  Tensor broadcast_a = Tensor::Broadcast(a, b.shape());

  std::shared_ptr<float> a_buffer(new float[b.size()], std::default_delete<float[]>());
  utils::BufferCopy(broadcast_a, a_buffer.get());

  std::shared_ptr<float> b_buffer(new float[b.size()], std::default_delete<float[]>());
  utils::BufferCopy(b, b_buffer.get());

  glas::axpy(b.size(), 1.0f, a_buffer.get(), b_buffer.get());

  utils::BufferAssign(b, b_buffer);
}

template <typename Op>
void binary_op_simd8(size_t N, const float* x, float* y, Op op) {
  for (size_t i = 0; i < (N - N % 8); i += 8) {
    __m256 x_vec = _mm256_loadu_ps(x + i);
    __m256 y_vec = _mm256_loadu_ps(y + i);
    y_vec = op(x_vec, y_vec);
    _mm256_storeu_ps(y + i, y_vec);
  }
}

Tensor binary_op(const Tensor& a, const Tensor& b, void (*op)(size_t, const float*, float*)) {
  std::vector<size_t> broadcast_shape = Tensor::GetBroadcastShape(a, b);
  size_t broadcast_size = std::accumulate(broadcast_shape.begin(), broadcast_shape.end(), 1, std::multiplies<size_t>());

  std::shared_ptr<float> a_buffer = utils::broadcast_tensor_to_buf(a, broadcast_shape, broadcast_size);
  std::shared_ptr<float> b_buffer = utils::broadcast_tensor_to_buf(b, broadcast_shape, broadcast_size);

  op(broadcast_size, a_buffer.get(), b_buffer.get());

  return Tensor(broadcast_shape, utils::compute_strides(broadcast_shape), 0, b_buffer);
}

void mul_simd(size_t N, const float* x, float* y) {
  binary_op_simd8(N, x, y, [] (__m256 x_vec, __m256 y_vec) { return _mm256_mul_ps(x_vec, y_vec); });
  for (size_t i = (N - N % 8); i < N; ++i) {
    y[i] *= x[i];
  }
}

Tensor mul(const Tensor& a, const Tensor& b) {
  return binary_op(a, b, mul_simd);
}

void div_simd(size_t N, const float* x, float* y) {
  binary_op_simd8(N, x, y, [] (__m256 x_vec, __m256 y_vec) { return _mm256_div_ps(x_vec, y_vec); });
  for (size_t i = (N - N % 8); i < N; ++i) {
    y[i] = x[i] / y[i];
  }
}

Tensor div(const Tensor& a, const Tensor& b) {
  return binary_op(a, b, div_simd);
}

void sub_simd(size_t N, const float* x, float* y) {
  binary_op_simd8(N, x, y, [] (__m256 x_vec, __m256 y_vec) { return _mm256_sub_ps(x_vec, y_vec); });
  for (size_t i = (N - N % 8); i < N; ++i) {
    y[i] = x[i] - y[i];
  }
}

Tensor sub(const Tensor& a, const Tensor& b) {
  return binary_op(a, b, sub_simd);
}

template <typename Op>
void unary_op_simd8(size_t N, float* y, float alpha, Op op) {
  const __m256 alpha_vec = _mm256_set1_ps(alpha);
  for (size_t i = 0; i < (N - N % 8); i += 8) {
    __m256 y_vec = _mm256_loadu_ps(y + i);
    y_vec = op(alpha_vec, y_vec);
    _mm256_storeu_ps(y + i, y_vec);
  }
}

Tensor unary_op(const Tensor& a, void (*op) (size_t, float*)) {
  std::shared_ptr<float> buffer = utils::broadcast_tensor_to_buf(a, a.shape(), a.size());

  op(a.size(), buffer.get());

  return Tensor(a.shape(), a.strides(), 0, buffer);
}

void neg_simd(size_t N, float* y) {
  unary_op_simd8(N, y, -1, [] (__m256 x_vec, __m256 y_vec) { return _mm256_mul_ps(x_vec, y_vec); });
  for (size_t i = (N - N % 8); i < N; ++i) {
    y[i] *= -1;
  }
}

Tensor neg(const Tensor& a) {
  return unary_op(a, neg_simd);
}

void mul_cons_simd(size_t N, float* y, float x) {
  unary_op_simd8(N, y, x, [] (__m256 x_vec, __m256 y_vec) { return _mm256_mul_ps(x_vec, y_vec); });
  for (size_t i = (N - N % 8); i < N; ++i) {
    y[i] *= x;
  }
}

void inv_simd(size_t N, float* y) {
  unary_op_simd8(N, y, 1, [] (__m256 x_vec, __m256 y_vec) { return _mm256_div_ps(x_vec, y_vec); });
  for (size_t i = (N - N % 8); i < N; ++i) {
    y[i] = 1 / y[i];
  }
}

Tensor inv(const Tensor& a) {
  return unary_op(a, inv_simd);
}

void log_buf(size_t N, float* y) {
  for (size_t i = 0; i < N; ++i) {
    y[i] = std::log(y[i]);
  }
}

Tensor log(const Tensor& a) {
  return unary_op(a, log_buf);
}

void exp_buf(size_t N, float* y) {
  for (size_t i = 0; i < N; ++i) {
    y[i] = std::exp(y[i]);
  }
}

Tensor exp(const Tensor& a) {
  return unary_op(a, exp_buf);
}

Tensor einsum(const Tensor& a, const Tensor& b, const std::string& equation) {
  // tokenize equation
  std::vector<std::string> tokens;
  std::string token;
  for (char c : equation) {
    if (isspace(c)) {
      if (token.size() != 0) tokens.push_back(token);
      token = "";
    } else if (c == ',') {
      if (token.size() != 0) tokens.push_back(token);
      tokens.push_back(",");
      token = "";
    } else {
      token += c;
    }
  }
  if(token.size() != 0) tokens.push_back(token);
  // group shapes
  std::map<std::string, size_t> a_shape_map;
  std::map<std::string, size_t> b_shape_map;
  std::map<std::string, size_t> c_shape_map;
  std::map<std::string, size_t> size_map;
  std::vector<size_t> c_shape;
  size_t index = 0;
  for (size_t i = 0, state = 0; i < tokens.size(); i++) {
    if (tokens[i] == ",") {
      if (state != 0) {
        throw std::invalid_argument("Invalid equation");
      }
      state = 1;
      index = 0;
    } else if (tokens[i] == "->") {
      if (state != 1) {
        throw std::invalid_argument("Invalid equation");
      }
      state = 2;
      index = 0;
    } else if (state == 0) {
      a_shape_map[tokens[i]] = index++;
      size_map[tokens[i]] = a.shape()[a_shape_map[tokens[i]]];
    } else if (state == 1) {
      b_shape_map[tokens[i]] = index++;
      size_map[tokens[i]] = b.shape()[b_shape_map[tokens[i]]];
    } else if (state == 2) {
      if (size_map.count(tokens[i]) == 0) {
        throw std::invalid_argument("Invalid equation");
      }
      c_shape_map[tokens[i]] = index++;
      c_shape.push_back(size_map[tokens[i]]);
    }
  }
  std::vector<std::string> all_dims;
  // check that the identical keys have identical values
  for (const auto& [key, value] : a_shape_map) {
    all_dims.push_back(key); // push back A
    if (b_shape_map.count(key) == 0) continue;
    if (b.shape()[b_shape_map[key]] != a.shape()[a_shape_map[key]]) {
      throw std::invalid_argument("Invalid equation");
    }
  }
  for (const auto& [key, value] : b_shape_map) {
    if (a_shape_map.count(key) == 0) {
      all_dims.push_back(key); // push back A - B to get A + B
      continue;
    }
    if (a.shape()[a_shape_map[key]] != b.shape()[b_shape_map[key]]) {
      throw std::invalid_argument("Invalid equation");
    }
  } 
  std::vector<int> c_strides(c_shape.size());
  size_t c_size = 1;
  for (int i = (int) c_shape.size() - 1; i >= 0; i--) {
    c_strides[i] = c_size;
    c_size *= c_shape[i];
  }
  // recursively multiply and sum
  std::shared_ptr<float> c_buffer(new float[c_size], std::default_delete<float[]>());
  std::fill(c_buffer.get(), c_buffer.get() + c_size, 0.0f);
  std::function<void(int, int, int, size_t)> recursive_einsum = [&](int a_offset, int b_offset, int c_offset, size_t N) {
    if (N == 0) {
      c_buffer.get()[c_offset] += a.data().get()[a_offset + a.offset()] * b.data().get()[b_offset + b.offset()];
    } else {
      std::string dim = all_dims[N - 1];
      for (size_t i = 0; i < size_map[dim]; i++) {
        int new_a_offset = a_shape_map.count(dim) ? a_offset + i * a.strides()[a_shape_map[dim]] : a_offset;
        int new_b_offset = b_shape_map.count(dim) ? b_offset + i * b.strides()[b_shape_map[dim]] : b_offset;
        int new_c_offset = c_shape_map.count(dim) ? c_offset + i * c_strides[c_shape_map[dim]] : c_offset;
        recursive_einsum(new_a_offset, new_b_offset, new_c_offset, N - 1);
      }
    }
  };
  recursive_einsum(0, 0, 0, all_dims.size());
  return Tensor(c_shape, c_strides, 0, c_buffer);
}

  void root_buf(size_t N, float* y) {
    for (size_t i = 0; i < N; ++i) {
      y[i] = std::sqrt(y[i]);
    }
  }

  Tensor root(const Tensor& a) {
    return unary_op(a, root_buf);
  }
Tensor reduce(const Tensor& a, std::function<float(float, float)> op, std::unordered_set<size_t> axes, float fill) {
  size_t size = 1;
  std::vector<size_t> shape;
  for (size_t i = 0; i < a.shape().size(); ++i) {
    if (axes.find(i) == axes.end()) {
      size *= a.shape()[i];
      shape.push_back(a.shape()[i]);
    }
  }

  std::shared_ptr<float> buffer(new float[size], std::default_delete<float[]>());
  std::fill(buffer.get(), buffer.get() + size, fill);

  std::function<void(size_t, int, int)> recursive_reduce = [a, op, axes, fill, buffer, size, recursive_reduce] (size_t depth, int buffer_offset, int tensor_offset) {
    if (depth == a.shape().size()) {
      buffer.get()[buffer_offset] = op(buffer.get()[buffer_offset], a.data().get()[a.offset() + tensor_offset]);
    }
    else {
      for (size_t i = 0; i < a.shape()[depth]; ++i) {
        int offset_inc = ((int) i) * a.strides()[depth];
        recursive_reduce(depth + 1, buffer_offset + (axes.find(i) != axes.end() ? 0 : offset_inc), tensor_offset + offset_inc);
      }
    }
  };
  recursive_reduce(0, 0, 0);

  return Tensor(shape, utils::compute_strides(shape), 0, buffer);
}
}



}