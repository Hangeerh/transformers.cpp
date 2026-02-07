#include "graph.hpp"
#include "graph_compiler.hpp"
#include <iostream>

void tr::CompiledGraph::feed(Node *node, const Tensor<float> &value) {
  if (!node->dst_edges.empty()) {
    tensor_store[node->dst_edges[0]->id] = value;
  }

  node->const_value = value;
}

tr::Tensor<float> tr::CompiledGraph::execute() {
  std::cout << "=== RUNNING COMPILE GRAPH ===" << std::endl;

  Tensor<float> result;

  for (const auto &step : steps) {
    std::cout << "  RUNNING: " << step.debug_name << "\n";

    std::vector<Tensor<float>> inputs;
    for (int edge_id : step.input_edge_ids) {
      if (tensor_store.count(edge_id)) {
        inputs.push_back(tensor_store[edge_id]);
      }
    }

    result = step.kernel(inputs, step.node);

    for (int edge_id : step.output_edge_ids) {
      tensor_store[edge_id] = result;
    }
  }
  return result;
}

tr::CompiledGraph tr::GraphCompiler::compile(tr::Node *sink) {
  tr::CompiledGraph compiled;

  std::vector<tr::Node *> sorted = tr::TopologicalSorter::sort(sink);

  std::unique_ptr<tr::ConstantFolding> fold =
      std::make_unique<tr::ConstantFolding>();
  fold->run(sorted);

  compiled.memory_plan = tr::MemoryPlanner::create_plan(sorted);

  for (Node *node : sorted) {
    ExecutionStep step;
    step.node = node;
    step.kernel = tr::Registry.get_kernel(node->type);
    step.debug_name = node->name;

    for (Edge *e : node->src_edges) {
      step.input_edge_ids.push_back(e->id);
    }
    for (Edge *e : node->dst_edges) {
      step.output_edge_ids.push_back(e->id);
    }

    // Track input/output nodes
    if (node->type == NodeType::SOURCE || node->type == NodeType::PLACEHOLDER) {
      compiled.input_nodes.push_back(node);
    }
    if (node->type == NodeType::SINK) {
      compiled.output_nodes.push_back(node);
    }

    compiled.steps.push_back(step);
  }
  return compiled;
}

void tr::OptimizationPass::run(NodeSet &nodes) {}
