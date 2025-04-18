#include <iostream>
#include "tensor.h"

int main() {
  std::vector<std::vector<float>> data = {{1, 2}, {3, 4}, {5, 6}};
  gooch::Tensor t = gooch::FromVector(data);
  std::cout << t.str() << std::endl;
  std::cout << gooch::FromVector(1.0f).str() << std::endl;
}