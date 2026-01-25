#pragma once

#include "compute_graph.hpp"
#include <cstddef>
namespace tr {

class Layer {};

class Dense : Layer {
public:
  Dense(size_t in_size, size_t out_size, bool bias);

  CGTensorNode *compile(CGTensorNode *in);

private:
  size_t m_in_size;
  size_t m_out_size;
  bool m_bias;
};

}; // namespace tr
