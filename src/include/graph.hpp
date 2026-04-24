#pragma once
#include "tensor.hpp"
#include <unordered_map>
#include <vector>

namespace tr {

class GraphNode;
class GraphValue;

class GraphNode {
  friend class Graph;

protected:
  int id = 0;
  // We need to know which ports the input values are fed into. This must be a
  // map in the node class because the same value can go into different ports of
  // different nodes.
  std::unordered_map<int, GraphValue *> input_values;
  // This needs to be a map to make writing to the value during inference easier
  std::unordered_map<int, GraphValue *> output_values;

public:
  virtual ~GraphNode() = 0;

  virtual std::string get_name() const = 0;
  // Infer output shapes of each output port
  virtual std::unordered_map<int, TensorShape> infer_output_shapes() const = 0;
  virtual bool validate() const = 0;
  virtual void forward() = 0;

  std::unordered_map<int, GraphValue *> &get_input_values() {
    return input_values;
  }
  std::unordered_map<int, GraphValue *> &get_output_values() {
    return output_values;
  }
};

struct GraphValue {
  TensorShape shape;
  // TODO: Should make unique_ptr
  Tensor<float> *tensor = nullptr;
  std::string name;
  GraphNode *producer = nullptr;
  std::vector<GraphNode *> consumers;

  // Certain nodes can return multiple values, so we need to distinguish between
  // them with a port number. For example MoE, we need to distinguish from
  // logits and aux loss.
  int id = 0;
  int output_port = 0;
  DType dtype = DType::Float32;

  enum class ValueType {
    TRANSIENT,
    INPUT,
    OUTPUT,
    PARAMETER,
  };

  ValueType type = ValueType::TRANSIENT;

  ~GraphValue();
};

// Currently binary x*W
class MatmulNode : public GraphNode {
  ~MatmulNode() override = default;

  std::string get_name() const override { return "MatmulNode"; }
  std::unordered_map<int, TensorShape> infer_output_shapes() const override;
  bool validate() const override;
  void forward() override;
};

// Currently binary x + b
class MatsumNode : public GraphNode {
  ~MatsumNode() override = default;

  std::string get_name() const override { return "MatsumNode"; }
  std::unordered_map<int, TensorShape> infer_output_shapes() const override;
  bool validate() const override;
  void forward() override;
};

class ReLUNode : public GraphNode {
  ~ReLUNode() override = default;

  std::string get_name() const override { return "ReLUNode"; }
  std::unordered_map<int, TensorShape> infer_output_shapes() const override;
  bool validate() const override;
  void forward() override;
};

class Graph {
private:
  std::vector<GraphNode *> all_nodes;
  std::vector<GraphValue *> all_values;
  std::vector<GraphValue *> graph_inputs;
  std::vector<GraphValue *> graph_outputs;
  std::vector<GraphNode *> sorted_nodes = {};
  int current_node_id = 0;
  int current_value_id = 0;
  bool frozen = false;

  bool validate_graph();

  void init_transient_and_output_tensors();

  std::vector<GraphNode *> topologic_sort();

public:
  Graph() = default;
  ~Graph();

  // Only allocs the value, does not initialize fields.
  GraphValue *alloc_value();

  // Allocs the node, but does not connect it with any values;
  template <typename T, typename... Args> T *alloc_node(Args &&...args) {
    T *node = new T(std::forward<Args>(args)...);
    node->id = current_node_id++;
    all_nodes.push_back(node);
    return node;
  }

  void connect_node_and_value(GraphNode *node, GraphValue *value, int port,
                              bool as_input);

  void register_as_input(GraphValue *value);
  void register_as_output(GraphValue *value);
  void register_as_parameter(GraphValue *value);

  // Should be done before freezing the graph
  void feed_parameter_value(GraphValue *value, Tensor<float> *parameter);

  bool freeze_graph();

  std::vector<Tensor<float> *> forward();
};
}; // namespace tr
