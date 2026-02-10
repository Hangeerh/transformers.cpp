#include "graph.hpp"

int main() {
  tr::Graph f;
  tr::Node *source = f.source(2, 4);
  tr::Node *layer1 = f.linear(source, 16, "layer1");
  tr::Node *layer2 = f.linear(layer1, 4, "layer2");
  tr::Node *sink = f.sink(layer2);
}
