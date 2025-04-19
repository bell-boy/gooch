#pragma once

#include "tensor.h"
#include <string>

namespace gooch {

Tensor reduce(Tensor& a, const std::string equation);


}