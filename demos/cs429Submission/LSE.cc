#include "tensor.h"
#include "glas.h"
#include "sgd.cc"
#include "random"

#include <iostream>
#include <climits>

int main(int argc  , char** argv) {
  if(argc!=4){
	  std::cout << "Expects 3 Arguments: Number of samples, classes, and iterations";
	  return 0;
  }
  size_t n = std::stoi(argv[1]) , c = std::stoi(argv[2]);
  gooch::Tensor weight = gooch::randn({n , c});
  std::vector<size_t> correct(n);
  for(int i = 0;i<(int)n;i++)
    correct[i] = rand() % c;
  gooch::Tensor loss({});
  std::vector<gooch::Tensor> params = {weight};
  gooch::SGD adam({params}, 0.1f);
  for (int i = 0; i <std::stoi(argv[3]); ++i) {
    loss = gooch::crossEntropyLoss(weight , correct);
    loss.Backward();
    std::cout << "Loss on " << (i+1) << "th iteration: " << *(loss.data().get()+loss.offset()) << "\n";
	std::cout << "Status on " << (i+1) << "th iteration: (Prediction , Correct):\n";
	for(int j = 0;j<(int)n;j++){
		float mx = INT_MIN;
		int pred = -1;
		for(int k = 0;k<(int)c;k++)if(*(weight(j)(k).data().get() + weight(j)(k).offset()) > mx)
			mx = *(weight(j)(k).data().get() + weight(j)(k).offset()) , pred = k;
		std::cout << j+1 << "th sample: " << pred << " " << correct[j] << "\n";
	}
	std::cout << "\n";
    /* std::cout << "Weights: " << weight.grad() << "\n"; */
    adam.step();
    weight.ZeroGrad();
    /* std::string d; */
    /* std::getline(std::cin , d); */
  }
  return 0;
}
