#include <iostream>
#include "tensor.h"
#include "einops.h"
#include <cassert>

int main() {
  gooch::Tensor t = gooch::zeros({100, 100, 100});
  float value = 0;
  for (int i = 0; i < (int) t.shape()[0]; i++) {
    for (int j = 0; j < (int) t.shape()[1]; j++) {
      for (int k = 0; k < (int) t.shape()[2]; k++) {
        gooch::View view = t(i, j, k);
        float* data = view.data().get() + view.offset();
        assert((*data - 0) < 1e-6);
        t(i, j, k) = gooch::FromVector(value++);
      }
    }
  }
  value = 0;
  for (int i = 0; i < (int) t.shape()[0]; i++) {
    for (int j = 0; j < (int) t.shape()[1]; j++) {
      for (int k = 0; k < (int) t.shape()[2]; k++) {
        gooch::View view = t(i, j, k);
        float* data = view.data().get() + view.offset();
        assert((*data - value++) < 1e-6);
      }
    }
  }
}