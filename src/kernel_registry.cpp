#include "kernel.hpp"
#include "tensor.hpp"
#include <stdexcept>

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

void tr::KernelRegistry::register_default_kernels() {
  register_kernel(NodeType::SOURCE,
                  [](const std::vector<Tensor<float>> &inputs,
                     Node *node) -> Tensor<float> {
                    return inputs.empty() ? node->const_value : inputs[0];
                  });

  register_kernel(
      NodeType::SINK,
      [](const std::vector<Tensor<float>> &inputs, Node *) -> Tensor<float> {
        return inputs.empty() ? Tensor<float>() : inputs[0];
      });

  register_kernel(NodeType::PLACEHOLDER,
                  [](const std::vector<Tensor<float>> &inputs,
                     Node *node) -> Tensor<float> {
                    if (!inputs.empty()) {
                      return inputs[0];
                    }

                    return node->const_value;
                  });

  // TODO make it more efficient?
  register_kernel(
      NodeType::SUM,
      [](const std::vector<Tensor<float>> &inputs, Node *) -> Tensor<float> {
        if (inputs.empty()) {
          throw std::runtime_error("Sum node has no inputs");
        }
        tr::Tensor<float> result =
            tr::Tensor<float>::zeroes_in_shape(inputs[0]);

        for (auto t : inputs) {
          result = tr::matsum(result, t);
        }

        return result;
      });

  register_kernel(
      NodeType::RELU,
      [](const std::vector<Tensor<float>> &inputs, Node *) -> Tensor<float> {
        if (inputs.empty())
          return Tensor<float>{};

        Tensor result = inputs[0];
        for (float &val : result.data()) {
          val = std::max(0.0f, val);
        }
        return result;
      });

  // TODO inplement matmul
  register_kernel(NodeType::MATMUL,
                  [](const std::vector<Tensor<float>> &inputs,
                     Node *) -> Tensor<float> { return tr::Tensor<float>(); });
}
