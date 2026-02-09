#pragma once
#include "tensor.hpp"
#include <unordered_map>

namespace tr {

struct Node;
struct Edge;

enum class NodeType {
  SOURCE,
  SINK,
  MATMUL,
  SUM,
  RELU,
  PLACEHOLDER,
  MUL,
  CONST
};

using EdgeSet = std::vector<Edge *>;
using NodeSet = std::vector<Node *>;

struct Node {
  int id;
  NodeType type;
  EdgeSet src_edges;
  EdgeSet dst_edges;
  std::string name;

  Tensor<float> const_value;
  std::unordered_map<std::string, int> int_attrs;
};

struct Edge {
  int id;
  Node *src_node;
  Node *dst_node;
  int src_port;
  int dst_port;
};

class Graph {
private:
  int current_node_id = 0;
  int current_edge_id = 0;
  NodeSet all_nodes;
  EdgeSet all_edges;

  Node *create_node(NodeType type);
  Edge *connect_nodes(Node *src, int src_port, Node *dst, int dst_port);

public:
  Graph() = default;
  ~Graph();

  Node *source();
  Node *sink(Node *n);
  Node *linear(Node *n);
};

}; // namespace tr
