#include "graph.hpp"
#include "tensor.hpp"
#include <algorithm>
#include <unordered_map>

namespace tr {

GraphValue::~GraphValue() { delete tensor; }

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
  assert(frozen);
  GraphValue *value = new GraphValue;
  value->id = current_value_id++;
  all_values.push_back(value);
  return value;
}

void Graph::connect_node_and_value(GraphNode *node, GraphValue *value, int port,
                                   bool as_input) {
  assert(frozen);
  if (as_input) {
    node->input_values[port] = value;
    value->consumers.push_back(node);
  } else {
    node->output_values[port] = value;
    value->producer = node;
    value->output_port = port;
  }
}

void Graph::register_as_input(GraphValue *value) {
  assert(frozen);
  graph_inputs.push_back(value);
  value->producer = nullptr;
  value->type = GraphValue::ValueType::INPUT;
}

void Graph::register_as_output(GraphValue *value) {
  assert(frozen);
  graph_outputs.push_back(value);
  value->type = GraphValue::ValueType::OUTPUT;
}

void Graph::register_as_parameter(GraphValue *value) {
  assert(frozen);
  value->type = GraphValue::ValueType::PARAMETER;
}

void Graph::feed_parameter_value(GraphValue *value, Tensor<float> *parameter) {
  assert(frozen);
  assert(value->type == GraphValue::ValueType::PARAMETER);

  // TODO: Check that the parameter's shape matches the expected shape specifiec in value

  value->tensor = parameter;
}

bool Graph::freeze_graph(){
  if(!validate_graph()){
    return false;
  }

  init_transient_and_output_tensors();

  sorted_nodes = topologic_sort();

  return true;
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
  if (input_values.size() != 2 || output_values.size() != 1) {
    return false;
  }

  TensorShape left = input_values.at(0)->shape;
  TensorShape right = input_values.at(1)->shape;
  TensorShape out = output_values.at(0)->shape;

  return TensorShape::is_valid_matrix_multiplication(left, right, out);
}

void MatmulNode::forward() {
  Tensor<float> *lhs = input_values.at(0)->tensor;
  Tensor<float> *rhs = input_values.at(1)->tensor;
  Tensor<float> *out = output_values.at(0)->tensor;
  matmul_in_place(lhs, rhs, out);
}

std::unordered_map<int, TensorShape> MatsumNode::infer_output_shapes() const {
  const GraphValue *lhs = input_values.at(0);
  const GraphValue *rhs = input_values.at(1);

  assert(lhs->shape == rhs->shape &&
         "MatsumNode requires matching input shapes");
  return {{0, lhs->shape}};
}

bool MatsumNode::validate() const {
  if (input_values.size() != 2 || output_values.size() != 1) {
    return false;
  }

  TensorShape left = input_values.at(0)->shape;
  TensorShape right = input_values.at(1)->shape;
  TensorShape out = output_values.at(0)->shape;

  if (left != right || right != out) {
    return false;
  }

  return true;
}

void MatsumNode::forward() {
  Tensor<float> *lhs = input_values.at(0)->tensor;
  Tensor<float> *rhs = input_values.at(1)->tensor;
  Tensor<float> *out = output_values.at(0)->tensor;
  matsum_in_place(lhs, rhs, out);
}

std::unordered_map<int, TensorShape> ReLUNode::infer_output_shapes() const {
  const GraphValue *input = input_values.at(0);
  return {{0, input->shape}};
}

bool ReLUNode::validate() const {
  if (input_values.size() != 1 || output_values.size() != 1) {
    return false;
  }

  TensorShape in = input_values.at(0)->shape;
  TensorShape out = output_values.at(0)->shape;

  if (in != out) {
    return false;
  }

  return true;
}

void ReLUNode::forward() {
  std::vector<float> &in = input_values.at(0)->tensor->data();
  std::vector<float> &out = output_values.at(0)->tensor->data();
  for (size_t i = 0; i < in.size(); i++) {
    out.at(i) = std::max(in.at(i), 0.0f);
  }
}

bool Graph::validate_graph() {
  assert(frozen);
  for (auto node : all_nodes) {
    if (!node->validate()) {
      return false;
    }
  }
  return true;
}

void Graph::init_transient_and_output_tensors() {
  assert(frozen);
  for (auto val : all_values) {
    if (val->type == GraphValue::ValueType::TRANSIENT ||
        val->type == GraphValue::ValueType::OUTPUT) {
      val->tensor = new Tensor<float>(val->shape);
    }
  }
}

std::vector<GraphNode *> Graph::topologic_sort() { assert(frozen); }
} // namespace tr
