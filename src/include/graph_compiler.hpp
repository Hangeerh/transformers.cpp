#pragma once
#include "graph.hpp"
#include "kernel.hpp"

namespace tr {
class TopologicalSorter {
public:
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

struct MemorySlot {
  int id;
  size_t size;
  int first_use;
  int last_use;
};

class MemoryPlanner {
public:
  struct Plan {
    std::unordered_map<int, int> edge_to_slot;
    std::vector<MemorySlot> slots;
    size_t total_memory;
  };

  static Plan create_plan(const std::vector<Node *> &sorted_nodes);
};

struct ExecutionStep {
  Node *node;
  KernelFn kernel;
  std::vector<int> input_edge_ids;
  std::vector<int> output_edge_ids;
  std::string debug_name;
};

class CompiledGraph {
public:
  std::vector<ExecutionStep> steps;
  std::vector<Node *> input_nodes;
  std::vector<Node *> output_nodes;
  MemoryPlanner::Plan memory_plan;
  std::unordered_map<int, Tensor<float>> tensor_store;

  void feed(Node *node, const Tensor<float> &value);

  Tensor<float> execute();
};

class GraphCompiler {
public:
  // TODO add options
  CompiledGraph compile(Node *sink);
};

} // namespace tr
