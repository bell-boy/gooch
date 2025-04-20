#include "einops.h"
#include "tensor.h"
#include <map>
namespace gooch {

Tensor reduce(Tensor& a, const std::string equation) {

  struct Index {
    std::string name;
    size_t index;
    bool is_reduced;
    size_t size;
  };

  std::vector<std::string> str_tokens;
  std::string token;
  for (char c : equation) {
    if (isspace(c)) {
      if (!token.empty()) {
        str_tokens.push_back(token);
        token = "";
      }
    } else {
      token += c;
    }
  }
  if (!token.empty()) {
    str_tokens.push_back(token);
  }

   bool valid = false; 
   std::map<std::string, Index> index_state;
   size_t index = 0;
   for (const auto& token : str_tokens) {
    if (token == "->") {
      if (valid) {
        throw std::invalid_argument("Invalid equation");
      }
      valid = true;
      continue;
    }

    if (!valid) {
      if (index_state.count(token)) {
        throw std::invalid_argument("Invalid equation");
      }
      size_t dim_index = index++;
      index_state[token] = {token, dim_index, true, a.shape()[dim_index]};
    } else {
      if (!index_state.count(token) || !index_state[token].is_reduced) {
        throw std::invalid_argument("Invalid equation");
      }
      index_state[token].is_reduced = false;
    }
   }
   std::vector<Index> indices(index_state.size());
   std::vector<size_t> new_shape;
   for (auto& [name, index] : index_state) {
    indices[index.index] = index;
    if (!index.is_reduced) {
      new_shape.push_back(index.size);
    }
   }
   Tensor result = zeros(new_shape);
   std::vector<Slice> result_slices;
   std::vector<Slice> a_slices;
   std::function<void(size_t)> recursive_reduce = [&](size_t i) {
    if (i == indices.size()) {
      View result_view = result[result_slices];
      View a_view = a[a_slices];
      result_view = result_view + a_view;
      return;
    }
    for (size_t j = 0; j < indices[i].size; ++j) {
      if (!indices[i].is_reduced) {
        result_slices.push_back(Slice(j));
      }
      a_slices.push_back(Slice(j));
      recursive_reduce(i + 1);
      if (!indices[i].is_reduced) {
        result_slices.pop_back();
      }
      a_slices.pop_back();
    }
   };
   recursive_reduce(0);
   return result;
  }

}

