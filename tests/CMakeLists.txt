function(new_test name file)
  add_executable(${name} ${file})
  target_link_libraries(${name} PRIVATE Transformers)
  add_test(NAME ${name} COMMAND ${name})
endfunction()


new_test(test_tensor_shape test_tensor_shape.cpp)
new_test(test_tensor_randint_from_shape test_tensor_randint_from_shape.cpp)
new_test(test_matmul test_matmul.cpp)
new_test(test_tensor_get_element test_tensor_get_element.cpp)
