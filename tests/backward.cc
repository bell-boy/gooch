#include "tensor.h"

int main() {
  gooch::Tensor a = gooch::FromVector(std::vector<std::vector<float>>{{1, 2}, {3, 4}}); 
  gooch::Tensor b = gooch::FromVector(std::vector<float>{3, 9});
  gooch::Tensor c = gooch::Einsum(a, b, "i j, i -> j");
  //std::cout << a << std::endl << b << std::endl << c << std::endl;
  c(0).Backward();
  return 0;
}