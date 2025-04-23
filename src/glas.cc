#include "glas.h"
#include "tensor.h"
#include "utils.h"

#include <immintrin.h>
#include <map>

namespace gooch {
namespace glas {

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
    y[i] = a * x[i] + y[i];
  }
}

Tensor add(const Tensor& a, const Tensor& b) {
  std::vector<size_t> broadcast_shape = Tensor::GetBroadcastShape(a, b);
  size_t size = std::accumulate(broadcast_shape.begin(), broadcast_shape.end(), 1, std::multiplies<size_t>());
  Tensor broadcast_a = Tensor::Broadcast(a, broadcast_shape);
  Tensor broadcast_b = Tensor::Broadcast(b, broadcast_shape);

  float* a_buffer = new float[size];
  std::shared_ptr<float> b_buffer(new float[size], std::default_delete<float[]>());
  utils::BufferCopy(broadcast_a, a_buffer);
  utils::BufferCopy(broadcast_b, b_buffer.get());

  glas::axpy(size, 1.0f, a_buffer, b_buffer.get());

  std::vector<int> strides(broadcast_shape.size());
  for (int i = (int) broadcast_shape.size() - 1, j = 1; i >= 0; i--) {
    strides[i] = j;
    j *= broadcast_shape[i];
  }
  return Tensor(broadcast_shape, strides, 0, b_buffer);
}

// in-place add, b += a
void add_(const Tensor& a, const Tensor& b) {
  Tensor broadcast_a = Tensor::Broadcast(a, b.shape());

  float* a_buffer = new float[b.size()];
  utils::BufferCopy(broadcast_a, a_buffer);

  float* b_buffer = new float[b.size()];
  utils::BufferCopy(b, b_buffer);

  glas::axpy(b.size(), 1.0f, a_buffer, b_buffer);
  delete[] a_buffer;

  utils::BufferAssign(b, b_buffer);
}

Tensor einsum(const std::string& equation, const Tensor& a, const Tensor& b) {
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
  tokens.push_back(token);
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

}
}
