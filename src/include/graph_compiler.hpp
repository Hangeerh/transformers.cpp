#pragma once
#include "graph.hpp"

namespace tr {
class TopologicalSorter {
  static NodeSet sort(Node *sink);
};

class OptimizationPass {
public:
  virtual ~OptimizationPass() = default;
  virtual void run(NodeSet &nodes);
  virtual std::string name() const = 0;
};

class ConstantFolding : OptimizationPass {
private:
  bool can_fold(Node *n);
  void fold(Node *n);

public:
  void run(NodeSet &nodes) override;
  std::string name() const override;
};
} // namespace tr
