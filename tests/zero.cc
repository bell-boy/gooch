#include <iostream>
#include "tensor.h"

int main() {
  gooch::Tensor t = gooch::zeros({3, 3, 3});
  std::cout << t.str() << std::endl;
  std::cout << t[{0, 2, 0}].str() << std::endl;
}