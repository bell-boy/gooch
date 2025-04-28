#include "tensor.h"
#include "glas.h"
#include "sgd.cc"
#include "random"

#include <iostream>

int main(int argc  , char** argv) {
  assert(argc==3);
  std::vector<float> x_data(100);
  std::vector<float> y_data(100);

  for (int i = 0; i < 100; ++i) {
    x_data[i] = i;
    y_data[i] = i * std::stoi(argv[1]);
  }

  gooch::Tensor noise = gooch::randn({100});
  gooch::Tensor x = gooch::FromVector(x_data);
  gooch::Tensor y = gooch::glas::add(gooch::FromVector(y_data), noise);
  gooch::Tensor m = gooch::randn({});
  gooch::Tensor loss_i({});
  gooch::Tensor loss({});

  std::vector<gooch::Tensor> params = {m};
  gooch::SGD adam({params}, 0.1f);
  for (int i = 0; i <std::stoi(argv[2]); ++i) {
    loss = gooch::zeros({});
    for (int j = 0; j < 100; ++j) {
      loss_i = y(j) - (x(j) * m);
      loss = loss + loss_i * loss_i;
    }
    loss = loss / gooch::FromVector(std::vector<float>{100.0f});
    loss.Backward();
    adam.step();
    std::cout << "Prediction: " <<  m << std::endl;
    std::cout << "Grad: " << m.grad() << "\n";
    m.ZeroGrad();
  }
  return 0;
}
