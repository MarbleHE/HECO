
# Without this, it does not find the MLIR functions. Is there a cleaner way than duplicating this from mlir/CMakeLists.txt?
find_package(MLIR REQUIRED CONFIG)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(MLIRDetectPythonEnv)
mlir_configure_python_dev_packages()

configure_file(setup.py.in setup.py.after_configure)
file(GENERATE OUTPUT setup.py INPUT ${CMAKE_CURRENT_BINARY_DIR}/setup.py.after_configure)

add_subdirectory(pyabc)

add_custom_target(python ALL DEPENDS pyabc)
add_custom_command(TARGET python
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/pyabc
    ${CMAKE_CURRENT_BINARY_DIR}/pyabc
)

install(CODE "execute_process(COMMAND ${Python3_EXECUTABLE} -m pip install ${CMAKE_BINARY_DIR}/python)")
