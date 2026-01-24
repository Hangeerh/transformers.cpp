#pragma once

#include "layer.hpp"
#include <vector>

namespace tr::model {

class Sequential {
public:
  Sequential();
  ~Sequential();

  void addLayer(tr::layer::Layer layer);

private:
  std::vector<std::unique_ptr<tr::layer::Layer>> layers;
};

}; // namespace tr::model
