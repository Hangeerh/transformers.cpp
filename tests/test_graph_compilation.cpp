#include "graph.hpp"
#include "graph_compiler.hpp"
#include "tensor.hpp"

void test_one_layer_dense() {
  tr::Graph g;
  tr::GraphCompiler compiler;

  tr::Node *source = g.source(1, 2);
  tr::Node *layer1 = g.dense(source, 2, "layer1");
  tr::Node *sink = g.sink(layer1);

  tr::Tensor<float> layer1W({2, 0, 0, 2}, {2, 2});
  tr::Tensor<float> layer1b({1, 0}, {1, 2});
  tr::Tensor<float> input({1, 1}, {1, 2});

  tr::CompiledGraph executable = compiler.compile(sink);

  executable.feed("layer1:W", layer1W);
  executable.feed("layer1:b", layer1b);
  executable.feed("source", input);

  tr::Tensor<float> result = executable.execute();

  assert(result.is_empty() == false);
}

void test_one_layer_linear_no_bias() {
  tr::Graph g;
  tr::GraphCompiler compiler;

  tr::Node *source = g.source(1, 2);
  tr::Node *layer1 = g.linear(source, 2, false, "layer1");
  tr::Node *sink = g.sink(layer1);

  tr::Tensor<float> layer1W({2, 0, 0, 2}, {2, 2});
  tr::Tensor<float> input({1, 1}, {1, 2});

  tr::CompiledGraph executable = compiler.compile(sink);

  executable.feed("layer1:W", layer1W);
  executable.feed("source", input);

  tr::Tensor<float> result = executable.execute();

  assert(result.is_empty() == false);
}

int main() {
  test_one_layer_linear_no_bias();
  test_one_layer_dense();
}
