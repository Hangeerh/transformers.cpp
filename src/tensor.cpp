#include "tensor.hpp"

tr::Tensor<int> tr::tensor_randint_in_shape(std::vector<size_t> shape,
                                            unsigned int seed = 1) {

  size_t count = 0;

  {
    size_t shape_size = shape.size();

    for (size_t index : shape) {
      assert(
          index > 0 &&
          "tr::tensor_randint_in_shape shape cannot have an index of value 0");

      count += index;
    }
  }

  srand(seed);

  std::vector<int> v(count);
  generate(v.begin(), v.end(), []() { return rand(); });

  srand(1);

  tr::Tensor<int> out = tr::tensor_from_data(v, shape);

  return out;
}
