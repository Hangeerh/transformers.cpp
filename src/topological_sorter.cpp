#include "graph_compiler.hpp"
#include <queue>
#include <unordered_set>

tr::NodeSet tr::TopologicalSorter::sort(Node *sink) {
  // Collect reachable nodes starting from SINK
  // We get all nodes that contribute to the output SINK
  std::unordered_set<Node *> reachable;
  std::function<void(Node *)> collect = [&](Node *n) {
    if (reachable.count(n)) {
      return;
    }
    reachable.insert(n);
    for (Edge *e : n->src_edges) {
      collect(e->src_node);
    }
  };
  collect(sink);

  // Compute in degrees
  std::unordered_map<Node *, int> in_degree;
  for (Node *n : reachable) {
    in_degree[n] = 0;
  }

  for (Node *n : reachable) {
    for (Edge *e : n->dst_edges) {
      if (reachable.count(e->dst_node)) {
        in_degree[e->dst_node]++;
      }
    }
  }

  // Process nodes starting with those without dependencies
  std::queue<Node *> ready;
  for (auto &[node, deg] : in_degree) {
    if (deg == 0) {
      ready.push(node);
    }
  }

  // Process nodes in order
  std::vector<Node *> sorted;
  while (!ready.empty()) {
    // Need to get the first element with front(), then remove it with pop()
    Node *n = ready.front();
    ready.pop();

    sorted.push_back(n);

    for (Edge *e : n->dst_edges) {
      Node *dst = e->dst_node;
      if (reachable.count(dst)) {
        in_degree[dst]--;
        if (in_degree[dst] == 0) {
          ready.push(dst);
        }
      }
    }
  }

  // Check for cycles
  if (sorted.size() != reachable.size()) {
    throw std::runtime_error("Cycle detected in graph");
  }

  return sorted;
}
