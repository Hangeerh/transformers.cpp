#include "graph.hpp"

tr::GraphBuilder::~GraphBuilder() {
  for (Node *n : all_nodes) {
    delete n;
  }

  for (Edge *e : all_edges) {
    delete e;
  }
}

tr::Node *tr::GraphBuilder::create_node(tr::NodeType type) {
  Node *node = new Node;
  node->type = type;
  node->id = current_node_id++;
  all_nodes.push_back(node);
  return node;
}

tr::Edge *tr::GraphBuilder::connect_nodes(tr::Node *src, int src_port,
                                          tr::Node *dst, int dst_port) {
  Edge *edge = new Edge;
  edge->id = current_edge_id++;
  all_edges.push_back(edge);

  edge->src_node = src;
  edge->src_port = src_port;

  edge->dst_node = dst;
  edge->dst_port = dst_port;

  src->dst_edges.push_back(edge);
  dst->src_edges.push_back(edge);

  return edge;
}

tr::Node *tr::GraphBuilder::source() {
  Node *source = create_node(tr::NodeType::SOURCE);
  return source;
}

tr::Node *tr::GraphBuilder::sink(Node *n) {
  Node *sink = create_node(tr::NodeType::SINK);
  connect_nodes(n, 0, sink, 0);
  return sink;
}

tr::Node *tr::GraphBuilder::linear(tr::Node *n) {
  Node *W = create_node(tr::NodeType::PLACEHOLDER);
  Node *mul = create_node(tr::NodeType::MATMUL);
  Node *b = create_node(tr::NodeType::PLACEHOLDER);
  Node *sum = create_node(tr::NodeType::SUM);
  Node *relu = create_node(tr::NodeType::RELU);

  // n*W
  connect_nodes(n, 0, mul, 0);
  connect_nodes(W, 0, mul, 1);

  // n*W + b
  connect_nodes(mul, 0, sum, 0);
  connect_nodes(b, 0, sum, 0);

  // ReLU(n*W + b)
  connect_nodes(sum, 0, relu, 0);

  return relu;
}
