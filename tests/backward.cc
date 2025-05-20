#include "tensor.h"

int main() {
  gooch::Tensor a = gooch::FromVector(std::vector<std::vector<float>>{{1, 2}, {3, 4}}); 
  gooch::Tensor b = gooch::FromVector(std::vector<float>{3, 9});
  gooch::Tensor c = gooch::Einsum(a, b, "i j, i -> j");
  std::cout << a << std::endl << b << std::endl << c << std::endl;
  c(0).Backward();
  std::cout << "grads:" << std::endl;
  std::cout << a.grad() << std::endl << b.grad() << std::endl << c.grad() << std::endl;
  
  gooch::Tensor d = gooch::randn({2, 2});
  std::cout << d << std::endl;
  std::cout << gooch::crossEntropyLoss(d, {0, 1}) << std::endl;
  return 0;
}