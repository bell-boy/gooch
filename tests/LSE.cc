#include "tensor.h"
#include "glas.h"
#include "sgd.cc"
#include "random"

#include <iostream>

int main(int argc  , char** argv) {
  assert(argc==2);
  size_t n = 2 , c = 3; // Can not increase c past 2, or n past 2 -> problem with loss.Backward();
  // issue is NOT with update_grad -> error is happening somewhere else ( guess is einsum or other function )
  gooch::Tensor weight = gooch::randn({n , c});
  std::vector<size_t> correct(n);
  for(int i = 0;i<(int)n;i++)
    correct[i] = rand() % c;
  gooch::Tensor loss({});
  std::vector<gooch::Tensor> params = {weight};
  gooch::SGD adam({params}, 0.1f);
  for (int i = 0; i <std::stoi(argv[1]); ++i) {
    loss = gooch::crossEntropyLoss(weight , correct);
    loss.Backward();
    // return 0;
    adam.step();
    for(int j = 0;j<(int)n;j++){
      std::cout << "Correct: " << correct[j] << "\n";
      std::cout << "Logits: " << weight(j) << "\n";
    }
    std::cout << "Loss: " << loss << "\n";
    weight.ZeroGrad();
    std::string d;
    std::getline(std::cin , d);
  }
  return 0;
}
