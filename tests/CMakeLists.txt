function(new_test name file)
  add_executable(${name} ${file})
  target_link_libraries(${name} PRIVATE Transformers)
  add_test(NAME ${name} COMMAND ${name})
endfunction()


# add_executable(test_tensor_shape test_tensor_shape.cpp)
# target_link_libraries(test_tensor_shape PRIVATE Transformers)
# add_test(NAME test_tensor_shape COMMAND test_tensor_shape)

new_test(test_tensor_shape test_tensor_shape.cpp)
new_test(test_tensor_randint_from_shape test_tensor_randint_from_shape.cpp)
