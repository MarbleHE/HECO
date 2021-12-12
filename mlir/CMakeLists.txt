cmake_minimum_required(VERSION 3.13.4)
project(abc-dialect LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

set(CMAKE_CXX_STANDARD 14 CACHE STRING "C++ standard to conform to")

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include)
add_subdirectory(lib)

####################################################################################
# Python Frontend
#
# Use pybind11 as C++ wrapper for the Python frontend
####################################################################################

# Install pyabc package (for dev)
find_package (Python3 COMPONENTS Interpreter Development)

if(NOT Python3_Interpreter_FOUND)
    message(WARNING "The ABC python frontend requires Python 3, but no installation
        was found. Skipping frontend building...")
endif()

# Set to test the python frontend. On by default.
if(NOT PYTHON_FRONTEND_DEV)
    set(PYTHON_FRONTEND_DEV true)
endif()

# Set to build the python frontend (installing pybind11 and building the wrapper). On by default.
if(NOT PYTHON_FRONTEND_BUILD)
    set(PYTHON_FRONTEND_BUILD true)
endif()

if(PYTHON_FRONTEND_BUILD)
    include(FetchContent)
    set(JSON_BuildTests OFF CACHE INTERNAL "")
    set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
    FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG v2.7.1)

    FetchContent_GetProperties(pybind11)
    if(NOT pybind11_POPULATED)
        FetchContent_Populate(pybind11)
        add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
    endif()
    add_subdirectory(python)
endif()

add_subdirectory(test)
add_subdirectory(abc-opt)
add_subdirectory(abc-translate)
add_subdirectory(fhe-opt)