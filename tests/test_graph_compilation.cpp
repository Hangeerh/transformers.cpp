#include "graph.hpp"
#include "graph_compiler.hpp"
#include "tensor.hpp"

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
  assert(result.data()[0] == 2);
  assert(result.data()[1] == 2);
}

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
  assert(result.data()[0] == 3);
  assert(result.data()[1] == 2);
}

void test_multiple_dense_layers() {
  tr::Graph g;
  tr::GraphCompiler compiler;

  tr::Node *source = g.source(1, 2);
  tr::Node *layer1 = g.dense(source, 3, "layer1");
  tr::Node *layer2 = g.dense(layer1, 3, "layer2");
  tr::Node *layer3 = g.dense(layer2, 2, "layer3");
  tr::Node *sink = g.sink(layer3);

  tr::Tensor<float> input({1, 1}, {1, 2});

  tr::Tensor<float> layer1W({1, 0, 1, 0, 1, 0}, {2, 3});
  tr::Tensor<float> layer1b({0, 0, 0}, {1, 3});

  tr::Tensor<float> layer2W({2, 0, 0, 0, 2, 0, 0, 0, 2}, {3, 3});
  tr::Tensor<float> layer2b({1, 1, 1}, {1, 3});

  tr::Tensor<float> layer3W({1, 0, 1, 0, 0, 1}, {3, 2});
  tr::Tensor<float> layer3b({0, -1}, {1, 2});

  tr::CompiledGraph executable = compiler.compile(sink);

  executable.feed("source", input);
  executable.feed("layer1:W", layer1W);
  executable.feed("layer1:b", layer1b);
  executable.feed("layer2:W", layer2W);
  executable.feed("layer2:b", layer2b);
  executable.feed("layer3:W", layer3W);
  executable.feed("layer3:b", layer3b);

  tr::Tensor<float> result = executable.execute();

  assert(result.is_empty() == false);
  assert(result.data()[0] == 6);
  assert(result.data()[1] == 2);
}

int main() {
  test_one_layer_linear_no_bias();
  test_one_layer_dense();
  test_multiple_dense_layers();
}
