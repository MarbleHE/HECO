# pybind11: add our wrapper for ABC
pybind11_add_module(pyabc abc_wrapper.cpp)
target_link_libraries(pyabc PRIVATE ast_opt)

set_target_properties(pyabc PROPERTIES OUTPUT_NAME _abc_wrapper)
