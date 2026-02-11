#include "graph.hpp"
#include "graph_compiler.hpp"
#include "tensor.hpp"

int main() {
  tr::Graph g;
  tr::GraphCompiler compiler;

  tr::Node *source = g.source(2, 4);
  tr::Node *layer1 = g.linear(source, 16, "layer1");
  tr::Node *sink = g.sink(layer1);

  tr::Tensor<float> layer1W({1, 0, 0, 1}, {2, 2});
  tr::Tensor<float> layer1b({1, 0}, {1, 2});
  tr::Tensor<float> input({0, 0}, {1, 2});

  tr::CompiledGraph executable = compiler.compile(sink);

  executable.feed("layer1:W", layer1W);
  executable.feed("layer1:b", layer1b);
  executable.feed("source", input);

  executable.execute();
}
