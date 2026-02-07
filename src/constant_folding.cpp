#include "graph_compiler.hpp"
#include "kernel.hpp"

bool tr::ConstantFolding::can_fold(tr::Node *n) {
  NodeType t = n->type;
  if (t == NodeType::CONST || t == NodeType::PLACEHOLDER ||
      t == NodeType::SOURCE || t == NodeType::SINK) {
    return false;
  }

  for (Edge *e : n->src_edges) {
    if (e->src_node->type != NodeType::CONST) {
      return false;
    }
  }
  return !n->src_edges.empty();
}

void tr::ConstantFolding::fold(tr::Node *n, tr::KernelRegistry *Registry) {
  std::vector<Tensor<float>> inputs;
  for (Edge *e : n->src_edges) {
    inputs.push_back(e->src_node->const_value);
  }

  auto kernel = Registry->get_kernel(n->type);
  n->const_value = kernel(inputs, n);

  n->type = NodeType::CONST;
  n->src_edges.clear();
}

void tr::ConstantFolding::run(NodeSet &nodes, tr::KernelRegistry *Registry) {
  int folded = 0;
  for (Node *n : nodes) {
    if (can_fold(n)) {
      fold(n, Registry);
      folded++;
    }
  }
}

std::string tr::ConstantFolding::name() const { return "ConstantFolding"; }
