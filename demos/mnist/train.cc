#include "tensor.h"
#include "bglu.h"
#include "mnist.h"
#include "sgd.cc"

#include <iostream>

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <path_to_mnist_csv>" << std::endl;
    return 1;
  }
  std::string path = argv[1];
  auto [labels, images] = GetMnist(path);
  gooch::Tensor x = gooch::FromVector(images);
  GatedLinearUnitMLP mlp(784, 100, 10);
  gooch::SGD sgd(mlp.params(), 1e-5f);
  int BATCH_SIZE = 10;
  for (int i = 0; i + BATCH_SIZE <  (int) labels.size(); i+=BATCH_SIZE) {
    std::vector<size_t> y_true;
    for (int j = i; j < BATCH_SIZE + i; j++) {
      y_true.push_back(labels[j]);
    }
    gooch::Tensor x_ = x(gooch::Slice(i,i+BATCH_SIZE-1));
    gooch::Tensor y_pred = mlp.forward(x_);

    gooch::Tensor loss = gooch::crossEntropyLoss(y_pred, y_true);

    std::cout << loss << std::endl;

    //mlp.ZeroGrad();
    loss.Backward(); 
    sgd.step();

    break;

  }
  return 0;
}
