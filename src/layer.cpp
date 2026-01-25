#include "transformers/layer.hpp"

tr::Dense::Dense(size_t in_size, size_t out_size, bool bias) {
  m_in_size = in_size;
  m_out_size = out_size;
  m_weights = tr::Tensor<float>({m_in_size, m_out_size});

  if (bias) {
    m_bias = tr::Tensor<float>({m_out_size, 1});
  }
}

tr::Tensor<float> tr::Dense::forward(tr::Tensor<float> &x) {
  tr::Tensor<float> out = matmul(m_weights, x);

  out = matsum(x, m_bias);

  return out;
}
