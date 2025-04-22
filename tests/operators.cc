#include "tensor.h"
#include <iostream>
#include <random>


int main() {
  std::default_random_engine generator;
  std::normal_distribution<float> d(0, 1);

  std::vector<std::vector<float>> a_vec;
  std::vector<std::vector<float>> b_vec;
  
  const int N = 1000, M = 1000;
  for (int i = 0; i < N; i++) {
    std::vector<float> a;
    for (int j = 0; j < M; j++) {
      a.push_back(d(generator));
    }
    a_vec.push_back(a);
  }

  for (int i = 0; i < N; i++) {
    std::vector<float> b;
    for (int j = 0; j < M; j++) {
      b.push_back(d(generator));
    }
    b_vec.push_back(b);
  }

  gooch::Tensor a = gooch::FromVector(a_vec);
  gooch::Tensor b = gooch::FromVector(b_vec);
  gooch::Tensor c = a + b;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      gooch::View view = c(i, j);
      float* data = view.data().get() + view.offset();
      assert(fabs(*data - a_vec[i][j] - b_vec[i][j]) < 1e-6);
    }
  }

  gooch::Tensor x = gooch::FromVector(1.0f);
  gooch::Tensor y = gooch::FromVector(std::vector<std::vector<float>>{{1, 2}, {3, 4}});
  gooch::Tensor z = x + y;
  z.grad()(gooch::Slice::all()) = gooch::FromVector(std::vector<std::vector<float>>{{1, 1}, {1, 1}});
  std::cout << z.grad() << std::endl;
  return 0;
}