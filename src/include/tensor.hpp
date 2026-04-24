#pragma once
#include <cassert>
#include <numeric>
#include <sstream>
#include <vector>

namespace tr {

enum class DType { Float32, Float64, Int32 };

struct TensorShape {
  std::vector<size_t> dims;

  TensorShape() = default;
  TensorShape(std::initializer_list<size_t> d) : dims(d) {}
  explicit TensorShape(std::vector<size_t> d) : dims(std::move(d)) {}

  size_t rank() const { return dims.size(); }

  size_t element_count() const {
    if (dims.empty())
      return 0;
    return std::accumulate(dims.begin(), dims.end(), size_t{1},
                           std::multiplies<size_t>());
  }

  size_t operator[](size_t i) const { return dims.at(i); }

  bool operator==(const TensorShape &other) const { return dims == other.dims; }
  bool operator!=(const TensorShape &other) const { return !(*this == other); }

  std::vector<size_t> to_vec() const { return dims; }

  std::string to_string() const {
    std::string s = "(";
    for (size_t i = 0; i < dims.size(); i++) {
      s += std::to_string(dims[i]);
      if (i + 1 < dims.size())
        s += ", ";
    }
    s += ")";
    return s;
  }

  static bool is_valid_matrix_multiplication(const TensorShape &left,
                                             const TensorShape &right,
                                             const TensorShape &out) {
    if (left.rank() != 2 || right.rank() != 2 || out.rank() != 2) {
      return false;
    }
    if (left.dims[1] != right.dims[0]) {
      return false;
    }
    if (out.dims[0] != left.dims[0] || out.dims[1] != right.dims[1]) {
      return false;
    }
    return true;
  }
};

template <typename T> class Tensor {
private:
  std::vector<T> tensor_data;
  std::vector<size_t> tensor_shape;
  std::vector<size_t> tensor_strides;

  void set_strides() {
    size_t last = tensor_shape.size() - 1;
    size_t stride = 1;

    for (size_t i = last; i > 0; --i) {
      stride *= tensor_shape.at(i);
      tensor_strides.at(i - 1) = stride;
    }

    tensor_strides.back() = 1;
  }

  size_t calculate_index(const std::vector<size_t> &indices) const {
    size_t index = 0;
    size_t count = indices.size();

    for (size_t i = 0; i < count; i++) {
      index += indices.at(i) * tensor_strides.at(i);
    }

    return index;
  }

public:
  Tensor() = default;

  Tensor(const std::vector<size_t> &shape) {
    tensor_shape = shape;
    tensor_strides = std::vector<size_t>(tensor_shape.size());

    set_strides();

    size_t total_elements = 1;
    for (size_t i : tensor_shape) {
      total_elements *= i;
    }

    tensor_data = std::vector<T>(total_elements);
  }

  Tensor(const std::vector<T> &data, const std::vector<size_t> &shape) {
    tensor_shape = shape;
    tensor_data = data;

    tensor_strides = std::vector<size_t>(tensor_shape.size());
    set_strides();
  }

  Tensor(const TensorShape &shape) : Tensor(shape.dims) {}

  const std::vector<size_t> shape() const { return tensor_shape; }

  std::vector<T> &data() { return tensor_data; }

  size_t dim() const { return tensor_shape.size(); }

  bool is_empty() { return tensor_data.empty(); }

  size_t total_elements() { return tensor_data.size(); }

  T &at(const std::vector<size_t> &indices) {

    size_t count = indices.size();

    assert(count == tensor_shape.size() &&
           "tr::Tensor::at dimension mismatch between tensor and indices");

    for (size_t i = 0; i < count; i++) {
      assert(indices.at(i) < tensor_shape.at(i) &&
             "tr::Tensor::at index out of bounds");
    }

    return tensor_data.at(calculate_index(indices));
  }

  const T &at(const std::vector<size_t> &indices) const {

    size_t count = indices.size();

    assert(count == tensor_shape.size() &&
           "tr::Tensor::at dimension mismatch between tensor and indices");

    for (size_t i = 0; i < count; i++) {
      assert(indices.at(i) < tensor_shape.at(i) &&
             "tr::Tensor::at index out of bounds");
    }

    return tensor_data.at(calculate_index(indices));
  }

  static Tensor<T> zeroes_in_shape(const Tensor<T> &tensor) {
    std::vector<size_t> shape = tensor.shape();
    return Tensor<T>(shape);
  }

  std::string display() {
    size_t rank = dim();
    if (rank > 2) {
      return std::string("Does not support tensors of rank > 2 yet");
    }

    std::stringstream ss;
    std::vector<size_t> cur_shape = shape();
    size_t height = cur_shape.at(0);
    size_t width = cur_shape.at(1);

    ss << "Tensor of shape (";
    for (size_t i = 0; i < rank - 1; i++) {
      ss << cur_shape.at(i) << ",";
    }
    ss << cur_shape.at(rank - 1) << ")\n";

    // Inefficient looping, will fix later
    ss << "{\n";
    for (size_t i = 0; i < height; i++) {
      ss << "  {";
      for (size_t j = 0; j < width - 1; j++) {
        ss << at({i, j}) << ", ";
      }
      ss << at({i, width - 1}) << "}\n";
    }
    ss << "}\n";

    return ss.str();
  }
};

Tensor<int> tensor_randint_in_shape(std::vector<size_t> shape,
                                    unsigned int seed);

template <typename T>
void matmul_in_place(Tensor<T> *t1, Tensor<T> *t2, Tensor<T> *out) {

  assert(t1->dim() == 2 && t2->dim() == 2 &&
         "tr::mat_mul() only accepts rank 2 tensors");

  assert(t1->shape().at(1) == t2->shape().at(0) &&
         "tr::mat_mul() dimension mismatch between multiplied matrices");

  assert(t1->shape()[0] == out->shape()[0]);
  assert(t2->shape()[1] == out->shape()[1]);

  size_t out_height = t1->shape().at(0);
  size_t out_width = t2->shape().at(1);
  size_t iters = t1->shape().at(1);

  for (size_t i = 0; i < out_height; i++) {
    for (size_t j = 0; j < out_width; j++) {
      T accumulator = 0;
      for (size_t k = 0; k < iters; k++) {
        accumulator += t1->at({i, k}) * t2->at({k, j});
      }
      out->at({i, j}) = accumulator;
    }
  }
}

template <typename T>
void matmul_in_place(const Tensor<T> &t1, const Tensor<T> &t2, Tensor<T> &out) {

  assert(t1.dim() == 2 && t2.dim() == 2 &&
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
template <typename T>
Tensor<T> matmul(const Tensor<T> &t1, const Tensor<T> &t2) {
  tr::Tensor<T> out({t1.shape().at(0), t2.shape().at(1)});
  matmul_in_place(t1, t2, out);

  return out;
}

template <typename T> Tensor<T> matmul(Tensor<T> *t1, Tensor<T> *t2) {
  tr::Tensor<T> out({t1->shape()[0], t2->shape()[1]});

  matmul_in_place(t1, t2, &out);

  return out;
}

template <typename T>
void matsum_in_place(Tensor<T> *t1, Tensor<T> *t2, Tensor<T> *out) {
  assert(t1->dim() == 2 && "tr::matsum only supports matrices");
  assert(t2->dim() == 2 && "tr::matsum only supports matrices");

  assert(t1->shape() == t2->shape() && "tr::matsum tensor shapes must match");
  assert(out->shape() == t1->shape() && "tr::matsum tensor shapes must match");

  std::vector<T> &t1_data = t1->data();
  std::vector<T> &t2_data = t2->data();
  std::vector<T> &out_data = out->data();

  size_t count = t1->total_elements();

  for (size_t i = 0; i < count; i++) {
    out_data.at(i) = t1_data.at(i) + t2_data.at(i);
  }
}

template <typename T> Tensor<T> matsum(Tensor<T> &t1, Tensor<T> &t2) {
  Tensor<T> out = tr::Tensor<T>::zeroes_in_shape(t1);

  matsum_in_place(&t1, &t2, &out);

  return out;
}

template <typename T> Tensor<T> matsum(Tensor<T> *t1, Tensor<T> *t2) {
  Tensor<T> out(t1->shape());

  matsum_in_place(t1, t2, out);

  return out;
}
} // namespace tr
