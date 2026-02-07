#include "graph_compiler.hpp"
#include <iostream>

tr::MemoryPlanner::Plan
tr::MemoryPlanner::create_plan(const std::vector<Node *> &sorted_nodes) {
  Plan plan;

  std::unordered_map<int, int> edge_first_use;
  std::unordered_map<int, int> edge_last_use;

  for (int step = 0; step < (int)sorted_nodes.size(); step++) {
    Node *n = sorted_nodes[step];

    for (Edge *e : n->dst_edges) {
      edge_first_use[e->id] = step;
    }

    for (Edge *e : n->src_edges) {
      edge_last_use[e->id] = step;
    }
  }

  // TODO implement interval scheduling
  int slot_id = 0;
  for (auto &[edge_id, first] : edge_first_use) {
    plan.edge_to_slot[edge_id] = slot_id++;
  }

  plan.total_memory = slot_id;

  std::cout << "[MEM] Allocated " << slot_id << " tensor slots\n";

  return plan;
}
