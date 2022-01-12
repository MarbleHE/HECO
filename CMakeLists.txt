cmake_minimum_required(VERSION 3.13.4)
project(abc-dialect LANGUAGES CXX C)

# Required for selecting MSVC runtime library
if (MSVC)
    cmake_policy(SET CMP0091 NEW)
endif ()

project(abc
        VERSION 0.1
        DESCRIPTION "Optimizer for Fully Homomorphic Encryption (FHE)"
        LANGUAGES CXX)

####################################################################################
## MLIR Main Setup
##
## Targets for the individual binaries are defined elsewhere (e.g. src/tools/<tool>)
####################################################################################
find_package(MLIR REQUIRED CONFIG)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/Modules")

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

add_subdirectory(include/abc/IR)
add_subdirectory(include/abc/Passes)
add_subdirectory(src/IR)
add_subdirectory(src/tools)

##############################
# Compiler Setup (esp. MSVC)
##############################

# Global flags aren't good CMAKE style, but these should work across pretty much all compilers
set(CXXFLAGS "--coverage --pedantic-errors -Wall -WError")

if (MSVC)
    message("Using MSVC")

    # Force dynamic MSVC runtime library on Windows
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

    # Switch MSVC into multi threaded mode
    add_compile_options(/MP)

    # Force MSVC to use slightly less broken template rules
    add_compile_options(/permissive-)

    # Force MSVC to conform to C++ standard's idea of exceptions
    add_compile_options(/EHsc)

    # Make (nearly) all warnings errors
    add_compile_options(/W4 /WX)

    # Try using variants
    add_compile_options(/std:c++17)

    # Don't force fopen_s (which is a MSVC-only thing)
    add_compile_definitions(_CRT_SECURE_NO_DEPRECATE)

    # Mark system headers as external
    # Treats all #include <..> as system includes, so requires some care from developers
    add_compile_options(/experimental:external /external:anglebrackets /external:W0)

endif ()

##############################
# Determine if SEAL is available
##############################

find_package(SEAL 3.6 CONFIG) # sets SEAL_FOUND variable

##############################
# Download and install nlohmann-json if required
##############################

find_package(nlohmann_json QUIET)
if (NOT nlohmann_json_FOUND)
    message("Downloading nlohmann_json")
    include(FetchContent)
    set(JSON_BuildTests OFF CACHE INTERNAL "")
    set(JSON_SystemInclude ON CACHE INTERNAL "")
    set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
    FetchContent_Declare(
            nlohmann_json
            GIT_REPOSITORY https://github.com/nlohmann/json.git
            GIT_TAG v3.7.3)
    FetchContent_MakeAvailable(nlohmann_json)
endif ()

##############################
# TARGET: cpp_ast
#
# Functionality around the legacy C++ ast
##############################

# All source files (*.cpp) for ABC
set(SOURCE_FILES
        ${abc_SOURCE_DIR}/src/ast/AbstractNode.cpp
        ${abc_SOURCE_DIR}/src/ast/Assignment.cpp
        ${abc_SOURCE_DIR}/src/ast/BinaryExpression.cpp
        ${abc_SOURCE_DIR}/src/ast/Block.cpp
        ${abc_SOURCE_DIR}/src/ast/Call.cpp
        ${abc_SOURCE_DIR}/src/ast/ExpressionList.cpp
        ${abc_SOURCE_DIR}/src/ast/For.cpp
        ${abc_SOURCE_DIR}/src/ast/Function.cpp
        ${abc_SOURCE_DIR}/src/ast/FunctionParameter.cpp
        ${abc_SOURCE_DIR}/src/ast/If.cpp
        ${abc_SOURCE_DIR}/src/ast/IndexAccess.cpp
        ${abc_SOURCE_DIR}/src/ast/OperatorExpression.cpp
        ${abc_SOURCE_DIR}/src/ast/Return.cpp
        ${abc_SOURCE_DIR}/src/ast/TernaryOperator.cpp
        ${abc_SOURCE_DIR}/src/ast/UnaryExpression.cpp
        ${abc_SOURCE_DIR}/src/ast/Variable.cpp
        ${abc_SOURCE_DIR}/src/ast/VariableDeclaration.cpp
        ${abc_SOURCE_DIR}/src/ast_parser/Errors.cpp
        ${abc_SOURCE_DIR}/src/ast_parser/File.cpp
        ${abc_SOURCE_DIR}/src/ast_utilities/Datatype.cpp
        ${abc_SOURCE_DIR}/src/ast_utilities/NodeUtils.cpp
        ${abc_SOURCE_DIR}/src/ast_utilities/Operator.cpp
        ${abc_SOURCE_DIR}/src/ast_utilities/ParentSettingVisitor.cpp
        ${abc_SOURCE_DIR}/src/ast_utilities/PlainVisitor.cpp
        ${abc_SOURCE_DIR}/src/ast_utilities/PrintVisitor.cpp
        ${abc_SOURCE_DIR}/src/ast_utilities/ProgramPrintVisitor.cpp
        ${abc_SOURCE_DIR}/src/ast_utilities/AbcAstToMlirVisitor.cpp
        ${abc_SOURCE_DIR}/src/ast_utilities/Scope.cpp
        ${abc_SOURCE_DIR}/src/ast_utilities/ScopedVisitor.cpp
        ${abc_SOURCE_DIR}/src/ast_parser/Parser.cpp
        ${abc_SOURCE_DIR}/src/ast_parser/PushBackStream.cpp
        ${abc_SOURCE_DIR}/src/ast_parser/Tokenizer.cpp
        ${abc_SOURCE_DIR}/src/ast_parser/Tokens.cpp
        )


# This contains all the header-only files (e.g., abstract classes, interfaces and templates that don't have *.cpp file)
set(HEADER_ONLY_SOURCES
        ${abc_SOURCE_DIR}/include/abc/ast/AbstractExpression.h
        ${abc_SOURCE_DIR}/include/abc/ast/AbstractStatement.h
        ${abc_SOURCE_DIR}/include/abc/ast/AbstractTarget.h
        ${abc_SOURCE_DIR}/include/abc/ast/Ast.h
        ${abc_SOURCE_DIR}/include/abc/ast_utilities/IVisitor.h
        ${abc_SOURCE_DIR}/include/abc/ast_utilities/VariableMap.h
        ${abc_SOURCE_DIR}/include/abc/ast_utilities/Visitor.h)

# This contains the headers that developers that want to link directly against ABC might need to have access to
# These are the headers that are installed when ABC is installed as a system-wide library
set(PUBLIC_HEADERS
        ${abc_SOURCE_DIR}/include/abc/ast/AbstractExpression.h
        ${abc_SOURCE_DIR}/include/abc/ast/AbstractNode.h
        ${abc_SOURCE_DIR}/include/abc/ast/AbstractStatement.h
        ${abc_SOURCE_DIR}/include/abc/ast/Assignment.h
        ${abc_SOURCE_DIR}/include/abc/ast/BinaryExpression.h
        ${abc_SOURCE_DIR}/include/abc/ast/Block.h
        ${abc_SOURCE_DIR}/include/abc/ast/Call.h
        ${abc_SOURCE_DIR}/include/abc/ast/ExpressionList.h
        ${abc_SOURCE_DIR}/include/abc/ast/For.h
        ${abc_SOURCE_DIR}/include/abc/ast/Function.h
        ${abc_SOURCE_DIR}/include/abc/ast/FunctionParameter.h
        ${abc_SOURCE_DIR}/include/abc/ast/If.h
        ${abc_SOURCE_DIR}/include/abc/ast/IndexAccess.h
        ${abc_SOURCE_DIR}/include/abc/ast/Literal.h
        ${abc_SOURCE_DIR}/include/abc/ast/OperatorExpression.h
        ${abc_SOURCE_DIR}/include/abc/ast/Return.h
        ${abc_SOURCE_DIR}/include/abc/ast/TernaryOperator.h
        ${abc_SOURCE_DIR}/include/abc/ast/UnaryExpression.h
        ${abc_SOURCE_DIR}/include/abc/ast/Variable.h
        ${abc_SOURCE_DIR}/include/abc/ast/VariableDeclaration.h
        ${abc_SOURCE_DIR}/include/abc/ast_parser/Errors.h
        ${abc_SOURCE_DIR}/include/abc/ast_parser/Parser.h
        ${abc_SOURCE_DIR}/include/abc/ast_parser/PushBackStream.h
        ${abc_SOURCE_DIR}/include/abc/ast_parser/Tokenizer.h
        ${abc_SOURCE_DIR}/include/abc/ast_parser/Tokens.h
        ${abc_SOURCE_DIR}/include/abc/ast_utilities/Datatype.h
        ${abc_SOURCE_DIR}/include/abc/ast_utilities/IVisitor.h
        ${abc_SOURCE_DIR}/include/abc/ast_utilities/NodeUtils.h
        ${abc_SOURCE_DIR}/include/abc/ast_utilities/Operator.h
        ${abc_SOURCE_DIR}/include/abc/ast_utilities/ParentSettingVisitor.h
        ${abc_SOURCE_DIR}/include/abc/ast_utilities/PlainVisitor.h
        ${abc_SOURCE_DIR}/include/abc/ast_utilities/PrintVisitor.h
        ${abc_SOURCE_DIR}/include/abc/ast_utilities/ProgramPrintVisitor.h
        ${abc_SOURCE_DIR}/include/abc/ast_utilities/Scope.h
        ${abc_SOURCE_DIR}/include/abc/ast_utilities/ScopedVisitor.h
        ${abc_SOURCE_DIR}/include/abc/ast_utilities/VariableMap.h
)

# build a library out of the code to be used by the tests
add_library(cpp_ast ${SOURCE_FILES} ${HEADER_ONLY_SOURCES})
add_library(abc::cpp_ast ALIAS cpp_ast)
target_link_libraries(cpp_ast PUBLIC nlohmann_json::nlohmann_json)
target_include_directories(cpp_ast PUBLIC ${PROJECT_SOURCE_DIR}/include)
set_target_properties(cpp_ast PROPERTIES PUBLIC_HEADER "${PUBLIC_HEADERS}")
set_target_properties(cpp_ast PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
target_compile_features(cpp_ast PUBLIC cxx_std_17)

if (SEAL_FOUND)
    message("Enabling SEAL functionality")
    target_include_directories(cpp_ast PUBLIC ${SEAL_INCLUDE_DIRS})
    target_link_libraries(cpp_ast PUBLIC SEAL::seal)
    target_compile_definitions(cpp_ast PUBLIC -DHAVE_SEAL_BFV)
endif (SEAL_FOUND)

#####################################################################################
### ast_opt installation
###
### ast_optimizer installed globally as a library
#####################################################################################
## Create a Version File
#include(CMakePackageConfigHelpers)
#write_basic_package_version_file(
#        ast_optConfigVersion.cmake
#        VERSION ${PACKAGE_VERSION}
#        COMPATIBILITY AnyNewerVersion
#)
#
## Configure (i.e. generate) the ast_optConfig.cmake file and copy both it and the version to install
#configure_file(ast_optConfig.cmake.in ast_optConfig.cmake @ONLY)
#install(FILES "${CMAKE_CURRENT_BINARY_DIR}/ast_optConfig.cmake"
#        "${CMAKE_CURRENT_BINARY_DIR}/ast_optConfigVersion.cmake"
#        DESTINATION lib/cmake/ast_opt)
#
#install(TARGETS ast_opt
#        EXPORT ast_optTargets
#        LIBRARY DESTINATION lib
#        ARCHIVE DESTINATION lib
#        RUNTIME DESTINATION bin
#        PUBLIC_HEADER DESTINATION include
#        INCLUDES DESTINATION include/ast_opt)
#
#install(DIRECTORY "include/"
#        DESTINATION "include")
#
#install(EXPORT ast_optTargets
#        FILE ast_optTargets.cmake
#        NAMESPACE ast_opt::
#        DESTINATION lib/cmake/ast_opt)

##############################
# TARGET: ast_demo
#
# A simple example of the ast_optimizer
##############################

add_executable(ast_demo
        ${abc_SOURCE_DIR}/examples/main.cpp)

target_link_libraries(ast_demo PRIVATE cpp_ast)


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

#####################################################################################
## TESTING
##
## only if this is the root project
#####################################################################################
if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    message("Enabling ABC Tests")
    enable_testing()
    add_subdirectory(test EXCLUDE_FROM_ALL)
endif ()

##############################
# TARGET: benchmarks
#
# A collection of benchmarks
##############################
#add_subdirectory(bench EXCLUDE_FROM_ALL)