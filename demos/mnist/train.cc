#include "tensor.h"
#include "bglu.h"
#include "mnist.h"

#include <iostream>

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <path_to_mnist_csv>" << std::endl;
    return 1;
  }
  std::string path = argv[1];
  auto [labels, images] = GetMnist(path);
  gooch::Tensor x = gooch::FromVector(images);
  std::cout << x(gooch::Slice(0, 1)) << std::endl;
  GatedLinearUnitMLP mlp(784, 100, 10);
  gooch::Tensor y_pred = mlp.forward(x(gooch::Slice(0, 1)));
  std::vector<size_t> y_true = {(size_t) labels[0], (size_t) labels[1]};
  std::cout << gooch::crossEntropyLoss(y_pred, y_true) << std::endl;
  std::cout << y_pred << "\n";
  return 0;
}
