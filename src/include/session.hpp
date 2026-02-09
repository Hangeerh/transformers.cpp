#pragma once
#include "graph.hpp"
#include "graph_compiler.hpp"
#include "kernel.hpp"

namespace tr {
class Session {
private:
  Graph graph;
  GraphCompiler compiler;
  KernelRegistry reg;

public:
  Session() = default;

  void init() { reg.register_default_kernels(); }
};
}; // namespace tr
