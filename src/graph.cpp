#include "graph.hpp"
#include "tensor.hpp"
#include <unordered_map>

namespace tr {

Graph::~Graph() {
  for (GraphNode *node : all_nodes) {
    delete node;
  }

  for (GraphValue *val : all_values) {
    delete val;
  }
}

GraphNode::~GraphNode() = default;

GraphValue *Graph::alloc_value() {
  GraphValue *value = new GraphValue;
  value->id = current_value_id++;
  all_values.push_back(value);
  return value;
}

void Graph::connect_node_and_value(GraphNode *node, GraphValue *value, int port,
                                   bool as_input) {
  if (as_input) {
    node->input_values[port] = value;
    value->consumers.push_back(node);
  } else {
    node->output_values.push_back(value);
    value->producer = node;
    value->output_port = port;
  }
}

std::unordered_map<int, TensorShape> MatmulNode::infer_output_shapes() const {
  const GraphValue *lhs = input_values.at(0);
  const GraphValue *rhs = input_values.at(1);

  assert(lhs->shape.rank() == 2 && rhs->shape.rank() == 2 &&
         "MatmulNode requires rank-2 inputs");
  assert(lhs->shape[1] == rhs->shape[0] &&
         "MatmulNode dimension mismatch for matrix multiplication");

  TensorShape output_shape{lhs->shape[0], rhs->shape[1]};
  return {{0, output_shape}};
}

bool MatmulNode::validate() const {
  if (input_values.size() != 2) {
    return false;
  }

  TensorShape left = input_values.at(0)->shape;
  TensorShape right = input_values.at(1)->shape;

  if (left.rank() != 2 || right.rank() != 2) {
    return false;
  }

  if (left.dims[1] != right.dims[0]) {
    return false;
  }

  return true;
}

std::unordered_map<int, TensorShape> MatsumNode::infer_output_shapes() const {
  const GraphValue *lhs = input_values.at(0);
  const GraphValue *rhs = input_values.at(1);

  assert(lhs->shape == rhs->shape &&
         "MatsumNode requires matching input shapes");
  return {{0, lhs->shape}};
}

bool MatsumNode::validate() const {
  if (input_values.size() != 2) {
    return false;
  }

  TensorShape left = input_values.at(0)->shape;
  TensorShape right = input_values.at(1)->shape;

  if (left.rank() != 2 || right.rank() != 2) {
    return false;
  }

  if (left.dims[1] != right.dims[1] || left.dims[0] != right.dims[0]) {
    return false;
  }

  return true;
}

std::unordered_map<int, TensorShape> ReLUNode::infer_output_shapes() const {
  const GraphValue *input = input_values.at(0);
  return {{0, input->shape}};
}

bool ReLUNode::validate() const {
  if (input_values.size() != 1) {
    return false;
  }

  return true;
}
} // namespace tr
