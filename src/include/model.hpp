#pragma once

#include "layer.hpp"
#include "tensor.hpp"
#include <vector>

namespace tr {

class Sequential {
public:
  Sequential();
  ~Sequential();

  void addLayer(Layer layer);
  Tensor<float> predict(Tensor<float> *x);

  void compile();

private:
  std::vector<std::unique_ptr<Layer>> layers;
};

}; // namespace tr
