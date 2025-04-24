#include "tensor.h"
#include "bglu.h"
#include <iostream>

int main() {
  GatedLinearUnitMLP mlp(784, 100, 10);
  gooch::Tensor x = gooch::randn({10, 784});
  gooch::Tensor y = mlp.forward(x);
  std::cout << y << std::endl;
  return 0;
}
