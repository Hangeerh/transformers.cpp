#pragma once
#include "graph.hpp"
#include <functional>

namespace tr {

using KernelFn =
    std::function<Tensor<float>(const std::vector<Tensor<float>> &, Node *)>;

class KernelRegistry {
private:
  std::unordered_map<NodeType, KernelFn> kernels;

public:
  void register_kernel(NodeType type, KernelFn fn);
  const KernelFn get_kernel(NodeType type);
  void register_default_kernels();
};
} // namespace tr
