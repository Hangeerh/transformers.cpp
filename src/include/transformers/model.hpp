#pragma once

#include "layer.hpp"
#include <vector>

namespace tr {

class Sequential {
public:
  Sequential();
  ~Sequential();

  void addLayer(Layer layer);

private:
  std::vector<std::unique_ptr<Layer>> layers;
};

}; // namespace tr
