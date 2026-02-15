#include "transformers/transformers.hpp"

int main() {
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
}
