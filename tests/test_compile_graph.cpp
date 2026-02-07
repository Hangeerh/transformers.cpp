#include "graph_compiler.hpp"

int main() {
  tr::GraphBuilder f;
  tr::Node *source = f.source();
  tr::Node *layer1 = f.linear(source);
  tr::Node *sink = f.sink(layer1);

  tr::GraphCompiler compiler;
  tr::CompiledGraph executable = compiler.compile(sink);
  executable.execute();
}
