#include "graph.hpp"
#include "graph_compiler.hpp"

int main() {
  tr::Graph g;
  tr::GraphCompiler compiler;

  tr::Node *source = g.source(2, 4);
  tr::Node *layer1 = g.linear(source, 16, "layer1");
  tr::Node *sink = g.sink(layer1);

  tr::CompiledGraph executable = compiler.compile(sink);
}
