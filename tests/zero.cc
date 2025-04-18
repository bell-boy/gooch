#include <iostream>
#include "tensor.h"

int main() {
  std::vector<std::vector<float>> data = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};
  gooch::Tensor t = gooch::FromVector(data);
  std::cout << t.str() << std::endl;
  std::cout << t[{gooch::Slice::all(), gooch::Slice(0, -1, 2)}].str() << std::endl;
  t[{gooch::Slice::all(), 0}] = gooch::FromVector(-1.0f);
  std::cout << t.str() << std::endl;
  std::cout << gooch::FromVector(1.0f).str() << std::endl;

}