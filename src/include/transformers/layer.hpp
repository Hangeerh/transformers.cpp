#pragma once

#include "transformers/tensor.hpp"
#include <cstddef>
namespace tr {

class Layer {
public:
  Layer();
  ~Layer();

private:
  void forward();
  void backward();
};

class Dense : Layer {
public:
  Dense(size_t in_size, size_t out_size, bool bias);
  ~Dense();

  tr::Tensor<float> forward(tr::Tensor<float> &x);

private:
  size_t m_in_size;
  size_t m_out_size;
  tr::Tensor<float> m_weights;
  tr::Tensor<float> m_bias;
};

}; // namespace tr
