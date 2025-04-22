#include "glas.h"
#include "tensor.h"

#include <immintrin.h>

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

void reduce(size_t N, const float* src, float* dest, int inc) {
  for (size_t i = 0; i < N; i++) {
    dest[0] += src[i * inc];
  }
}

}
}
