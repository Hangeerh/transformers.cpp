#include "graph.hpp"

int main() {
  tr::GraphBuilder f;
  tr::Node *source = f.source();
  tr::Node *layer1 = f.linear(source);
  tr::Node *layer2 = f.linear(layer1);
  tr::Node *sink = f.sink(layer2);
}
