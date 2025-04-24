#include "tensor.h"
#include <iostream>
#include <random>


int main() {
  std::default_random_engine generator;
  std::normal_distribution<float> d(0, 1);

  std::vector<std::vector<float>> a_vec;
  std::vector<std::vector<float>> b_vec;
  
  const int N = 5, M = 5;
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

  c = a * b;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      gooch::View view = c(i, j);
      float* data = view.data().get() + view.offset();
      assert(fabs(*data - a_vec[i][j] * b_vec[i][j]) < 1e-6);
    }
  }

  c = a / b;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      gooch::View view = c(i, j);
      float* data = view.data().get() + view.offset();
      assert(fabs(*data - (a_vec[i][j] / b_vec[i][j])) < 1e-6);
    }
  }

  c = a - b;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      gooch::View view = c(i, j);
      float* data = view.data().get() + view.offset();
      assert(fabs(*data - a_vec[i][j] + b_vec[i][j]) < 1e-6);
    }
  }

  // Test broadcasting
  // broadcasting bug, edge case when broadcasting from 1
  gooch::Tensor x = gooch::randn({N, 1});
  gooch::Tensor y = gooch::randn({M});
  c = a + x;
  std::cout << "a + x " <<  c <<  std::endl;
  std::cout << "a " << a << std::endl;
  std::cout << "x " << x << std::endl;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      gooch::View view = c(i, j);
      gooch::View x_view = x(i, 0);
      gooch::View a_view = a(i, j);
      float* view_data = view.data().get() + view.offset();
      float* x_data = x_view.data().get() + x_view.offset();
      float* a_data = a_view.data().get() + a_view.offset();
      std::cout << *view_data << " " << *x_data +  *a_data << " " << *x_data << " " << *a_data << std::endl;
      assert(fabs(*view_data - *x_data - *a_data) < 1e-6);
    }
  }
  return 0;
}