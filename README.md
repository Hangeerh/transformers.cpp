# transformers.cpp
A demonstrative c++ deep-learning library

## Usage

```cpp
#include "transformers/transformers.hpp"
#include <iostream>

int main() {
  // Create a graph g
  tr::Graph g;

  // Create the graph compiler that we will use to compile the graph g
  tr::GraphCompiler compiler;

  // Create a source node.
  // The source node is where we feed in our input
  // We specify that our input tensor should be 1 rows x 2 columns
  tr::Node *source = g.source(1, 2);

  // Create a dense layer ReLU(x@W + b), with inputs feeding in from the sink
  // node. We specify that the output dimension should be a row vector of
  // dimension 2. We also specify a human readable indentifier "layer1"
  tr::Node *layer1 = g.dense(source, 2, "layer1");

  // Create the output sink node from layer1
  // The sink node is where we get our output tensor
  tr::Node *sink = g.sink(layer1);

  // Create weights and biases for the dense layer
  // Create a weights matrix
  // [ 2 0 ]
  // [ 0 2 ]
  tr::Tensor<float> layer1W({2, 0, 0, 2}, {2, 2});

  // Create a bias vector
  // [1 0]
  tr::Tensor<float> layer1b({1, 0}, {1, 2});

  // Create our input tensor
  // The first parameter is the data, and the second is the shape
  // The shape is row major, so we create a 1x2 row vector [1,1]
  tr::Tensor<float> input({1, 1}, {1, 2});

  // Compile the graph
  tr::CompiledGraph executable = compiler.compile(sink);

  // Load the weights and input into the compiled graph
  executable.feed("layer1:W", layer1W);
  executable.feed("layer1:b", layer1b);
  executable.feed("source", input);

  // Run the compiled graph
  // First runs x@W
  // [1 1] [2 0] = [2 2]
  //       [0 2]
  //
  // Then +b
  // [2 2] + [1 0] = [3 2]
  //
  // Then ReLU
  // ReLU([3 2]) = [3 2]
  //
  // We get our output [3 2]
  tr::Tensor<float> result = executable.execute();

  // Display the output tensor
  std::cout << result.display() << std::endl;
}
```

## Building from source

Build using cmake
```shell
mkdir build
cd build
cmake ..
make
```
