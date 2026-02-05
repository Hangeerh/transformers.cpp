#include "kernel.hpp"
#include "tensor.hpp"

void tr::KernelRegistry::register_kernel(NodeType type, KernelFn fn) {
  kernels[type] = std::move(fn);
}

const tr::KernelFn tr::KernelRegistry::get_kernel(NodeType type) {
  auto it = kernels.find(type);
  if (it != kernels.end()) {
    return it->second;
  }
  throw std::runtime_error("No kernel registered for this operation");
}

void tr::register_default_kernels() {
  Registry.register_kernel(
      NodeType::SOURCE,
      [](const std::vector<Tensor<float>> &inputs, Node *) -> Tensor<float> {
        return inputs.empty() ? Tensor<float>() : inputs[0];
      });

  Registry.register_kernel(
      NodeType::SINK,
      [](const std::vector<Tensor<float>> &inputs, Node *) -> Tensor<float> {
        return inputs.empty() ? Tensor<float>() : inputs[0];
      });

  Registry.register_kernel(NodeType::PLACEHOLDER,
                           [](const std::vector<Tensor<float>> &inputs,
                              Node *node) -> Tensor<float> {
                             if (!inputs.empty()) {
                               return inputs[0];
                             }

                             return node->const_value;
                           });

  // TODO tensor sum
  Registry.register_kernel(
      NodeType::SUM,
      [](const std::vector<Tensor<float>> &inputs, Node *) -> Tensor<float> {
        Tensor<float> result;
        return result;
      });

  Registry.register_kernel(
      NodeType::RELU,
      [](const std::vector<Tensor<float>> &inputs, Node *) -> Tensor<float> {
        if (inputs.empty())
          return Tensor<float>{};

        Tensor result = inputs[0];
        for (float &val : *result.get_data_handle()) {
          val = std::max(0.0f, val);
        }
        return result;
      });

  // TODO matmul kernel
  Registry.register_kernel(
      NodeType::MATMUL,
      [](const std::vector<Tensor<float>> &inputs, Node *) -> Tensor<float> {
        return Tensor<float>{};
      });
}
