#include "tensor.h"
#include "bglu.h"

GatedLinearUnitMLP::GatedLinearUnitMLP(size_t input_dim, size_t hidden_dim, size_t output_dim) : W_1_(gooch::randn({hidden_dim, input_dim})), W_2_(gooch::randn({hidden_dim, input_dim})), W_down_(gooch::randn({output_dim, hidden_dim})) {}

gooch::Tensor GatedLinearUnitMLP::forward(gooch::Tensor input_batch) {
  gooch::Tensor x_1 = gooch::Einsum(input_batch, W_1_, "batch in_dim, hidden_dim input_dim -> batch hidden_dim");
  gooch::Tensor x_2 = gooch::Einsum(input_batch, W_2_, "batch in_dim, hidden_dim input_dim -> batch hidden_dim");
  return gooch::Einsum((x_1 * x_2) / gooch::FromVector(3136.0f), W_down_, "batch hidden_dim, output_dim hidden_dim -> batch output_dim");
}