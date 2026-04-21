#pragma once
#include "tensor.hpp"
#include <cassert>
#include <vector>

namespace tr {

class TensorView {
private:
  float *data_;
  TensorShape shape_;
  std::vector<size_t> strides_;
  size_t offset_;

  void compute_strides() {
    strides_.resize(shape_.rank());
    if (shape_.rank() == 0)
      return;
    strides_.back() = 1;
    for (int i = (int)shape_.rank() - 2; i >= 0; --i) {
      strides_[i] = strides_[i + 1] * shape_.dims[i + 1];
    }
  }

  size_t flat_index(const std::vector<size_t> &indices) const {
    assert(indices.size() == shape_.rank());
    size_t idx = 0;
    for (size_t i = 0; i < indices.size(); i++) {
      assert(indices[i] < shape_.dims[i]);
      idx += indices[i] * strides_[i];
    }
    return idx;
  }

public:
  explicit TensorView(Tensor<float> &tensor)
      : data_(tensor.data().data()), shape_(TensorShape(tensor.shape())),
        offset_(0) {
    compute_strides();
  }

  TensorView(float *data, TensorShape shape, std::vector<size_t> strides,
             size_t offset = 0)
      : data_(data), shape_(std::move(shape)), strides_(std::move(strides)),
        offset_(offset) {}

  TensorView(float *data, TensorShape shape, size_t offset = 0)
      : data_(data), shape_(std::move(shape)), offset_(offset) {
    compute_strides();
  }

  float *data() { return data_ + offset_; }
  const float *data() const { return data_ + offset_; }

  const TensorShape &shape() const { return shape_; }
  size_t rank() const { return shape_.rank(); }
  size_t element_count() const { return shape_.element_count(); }
  const std::vector<size_t> &strides() const { return strides_; }

  float &at(const std::vector<size_t> &indices) {
    return data_[offset_ + flat_index(indices)];
  }

  const float &at(const std::vector<size_t> &indices) const {
    return data_[offset_ + flat_index(indices)];
  }

  TensorView slice(int dim, size_t start, size_t end) const {
    assert(dim >= 0 && dim < (int)shape_.rank());
    assert(start < end && end <= shape_.dims[dim]);

    TensorShape new_shape = shape_;
    new_shape.dims[dim] = end - start;

    size_t new_offset = offset_ + start * strides_[dim];

    return TensorView(data_, std::move(new_shape), strides_, new_offset);
  }
};

} // namespace tr
