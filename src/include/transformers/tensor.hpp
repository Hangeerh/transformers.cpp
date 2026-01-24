#pragma once
#include <cassert>
#include <vector>

namespace tr {

template <typename T> class Tensor {
private:
  std::vector<T> m_data;
  std::vector<size_t> m_shape;
  std::vector<size_t> m_strides;

  void set_strides() {
    size_t last = m_shape.size() - 1;
    size_t stride = 1;

    for (size_t i = last; i > 0; --i) {
      stride *= m_shape.at(i);
      m_strides.at(i - 1) = stride;
    }

    m_strides.back() = 1;
  }

  size_t calculate_index(const std::vector<size_t> &indices) {
    size_t index = 0;
    size_t count = indices.size();

    for (size_t i = 0; i < count; i++) {
      index += indices.at(i) * m_strides.at(i);
    }

    return index;
  }

public:
  Tensor(const std::vector<size_t> &shape) {
    m_shape = shape;
    m_strides = std::vector<size_t>(m_shape.size());

    set_strides();

    size_t total_elements = 1;
    for (size_t i : m_shape) {
      total_elements *= i;
    }

    m_data = std::vector<T>(total_elements);
  }

  Tensor() {
    m_data = {(T)0};
    m_shape = {1};
    m_strides = {1};
  }

  const std::vector<size_t> shape() {
    return (const std::vector<size_t>)m_shape;
  }

  T &at(const std::vector<size_t> &indices) {

    size_t count = indices.size();

    assert(count == m_shape.size() &&
           "tr::Tensor::at dimension mismatch between tensor and indices");

    for (size_t i = 0; i < count; i++) {
      assert(indices.at(i) < m_shape.at(i) &&
             "tr::Tensor::at index out of bounds");
    }

    return m_data.at(calculate_index(indices));
  }

  std::vector<T> get_data() { return m_data; }

  std::vector<T> *get_data_handle() { return &m_data; }
};

template <typename T>
Tensor<T> tensor_from_data(std::vector<T> data, std::vector<size_t> shape) {

  {
    size_t count = 0;
    size_t shape_size = shape.size();

    for (size_t i = 0; i < shape_size; i++) {
      size_t index = shape.at(i);
      assert(index > 0 &&
             "tr::tensor_from_data shape cannot have an index of value 0");
      count += index;
    }

    assert(count == data.size() &&
           "tr::tensor_from_data size of data does not match tensor shape");
  }

  Tensor<T> out(shape);

  *out.get_data_handle() = data;

  return out;
}

Tensor<int> tensor_randint_in_shape(std::vector<size_t> shape,
                                    unsigned int seed);

template <typename T>
void matmul_in_place(Tensor<T> &t1, Tensor<T> &t2, Tensor<T> &out) {

  assert(t1.shape().size() == 2 && t2.shape().size() == 2 &&
         "tr::mat_mul() only accepts rank 2 tensors");

  assert(t1.shape().at(1) == t2.shape().at(0) &&
         "tr::mat_mul() dimension mismatch between multiplied matrices");

  assert(t1.shape()[0] == out.shape()[0]);
  assert(t2.shape()[1] == out.shape()[1]);

  size_t out_height = t1.shape().at(0);
  size_t out_width = t2.shape().at(1);
  size_t iters = t1.shape().at(1);

  for (size_t i = 0; i < out_height; i++) {
    for (size_t j = 0; j < out_width; j++) {
      T accumulator = 0;
      for (size_t k = 0; k < iters; k++) {
        accumulator += t1.at({i, k}) * t2.at({k, j});
      }
      out.at({i, j}) = accumulator;
    }
  }
}

template <typename T> Tensor<T> matmul(Tensor<T> &t1, Tensor<T> &t2) {
  tr::Tensor<T> out({t1.shape()[0], t2.shape()[1]});

  matmul_in_place(t1, t2, out);

  return out;
}

template <typename T>
void matsum_in_place(Tensor<T> &t1, Tensor<T> &t2, Tensor<T> &out) {
  assert(t1.shape().size() == 2 && "tr::matsum only supports matrices");
  assert(t2.shape().size() == 2 && "tr::matsum only supports matrices");

  assert(t1.shape() == t2.shape() && "tr::matsum tensor shapes must match");
  assert(out.shape() == t1.shape() && "tr::matsum tensor shapes must match");

  std::vector<T> *t1_data = t1.get_data_handle();
  std::vector<T> *t2_data = t2.get_data_handle();
  std::vector<T> *out_data = out.get_data_handle();

  size_t count = t1_data->size();

  for (size_t i = 0; i < count; i++) {
    out_data->at(i) = t1_data->at(i) + t2_data->at(i);
  }
}

template <typename T> Tensor<T> matsum(Tensor<T> &t1, Tensor<T> &t2) {
  Tensor<T> out(t1.shape());

  matsum_in_place(t1, t2, out);

  return out;
}
} // namespace tr
