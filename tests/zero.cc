#include <iostream>
#include "tensor.h"
#include "einops.h"
#include <cassert>

int main() {
  gooch::Tensor t = gooch::zeros({100, 100, 100});
  /*
  float value = 0;
  for (size_t i = 0; i < t.shape()[0]; i++) {
    for (size_t j = 0; j < t.shape()[1]; j++) {
      for (size_t k = 0; k < t.shape()[2]; k++) {
        gooch::View view = t[{i, j, k}];
        float* data = view.data().get() + view.offset();
        assert((*data == 0));
        t[{i, j, k}] = gooch::FromVector(value++);
      }
    }
  }
  value = 0;
  for (size_t i = 0; i < t.shape()[0]; i++) {
    for (size_t j = 0; j < t.shape()[1]; j++) {
      for (size_t k = 0; k < t.shape()[2]; k++) {
        gooch::View view = t[{i, j, k}];
        float* data = view.data().get() + view.offset();
        assert((*data == value++));
      }
    }
  }
  */
}