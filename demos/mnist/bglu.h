#include "tensor.h"


class GatedLinearUnitMLP {
private:
  gooch::Tensor W_1_;
  gooch::Tensor W_2_;
  gooch::Tensor W_down_;
public:
  GatedLinearUnitMLP(size_t input_dim, size_t hidden_dim, size_t output_dim);
  gooch::Tensor forward(gooch::Tensor input_batch);
};