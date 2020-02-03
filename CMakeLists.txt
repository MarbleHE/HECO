cmake_minimum_required(VERSION 3.14)
set(CMAKE_CXX_STANDARD 17)
project(ast_optimizer)

##############################
# Download GoogleTest framework
##############################

# Download and unpack googletest at configure time
set_property(GLOBAL PROPERTY CTEST_TARGETS_ADDED 1)
configure_file(${PROJECT_SOURCE_DIR}/libs/googletest.in googletest-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download)
if (result)
    message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif ()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download)
if (result)
    message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif ()

# Prevent overriding the parent project's compiler/linker settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest to our build as an external target
execute_process(COMMAND ${CMAKE_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/googletest-src -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/googletest-install WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-build)
execute_process(COMMAND ${CMAKE_EXECUTABLE} --build ${CMAKE_CURRENT_BINARY_DIR}/googletest-build --target install)

# The gtest/gtest_main targets carry header search path dependencies automatically when using CMake 2.8.11 or later.
# Otherwise we have to add them here ourselves.
if (CMAKE_VERSION VERSION_LESS 2.8.11)
    include_directories("${gtest_SOURCE_DIR}/include")
endif ()

##############################
# Download nlohmann-json
##############################

# exclude the tests from the build
set(JSON_BuildTests OFF CACHE INTERNAL "")

configure_file(libs/nlohmann-json.in nlohmann-json-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/nlohmann-json-download)
if (result)
    message(FATAL_ERROR "CMake step for nlohmann-json failed: ${result}")
endif ()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/nlohmann-json-download)
if (result)
    message(FATAL_ERROR "Build step for nlohmann failed: ${result}")
endif ()
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/nlohmann-json-src
        ${CMAKE_CURRENT_BINARY_DIR}/nlohmann-json-build
        EXCLUDE_FROM_ALL)

##############################
# TARGET: ast_optimizer
##############################

include_directories(
        ${ast_optimizer_SOURCE_DIR}/include/ast
        ${ast_optimizer_SOURCE_DIR}/include/optimizer
        ${ast_optimizer_SOURCE_DIR}/examples)

set(SOURCE_FILES
        ${ast_optimizer_SOURCE_DIR}/src/ast/AbstractExpr.cpp
        ${ast_optimizer_SOURCE_DIR}/src/ast/AbstractStatement.cpp
        ${ast_optimizer_SOURCE_DIR}/src/ast/Variable.cpp
        ${ast_optimizer_SOURCE_DIR}/src/ast/Literal.cpp
        ${ast_optimizer_SOURCE_DIR}/src/ast/LiteralBool.cpp
        ${ast_optimizer_SOURCE_DIR}/src/ast/LiteralInt.cpp
        ${ast_optimizer_SOURCE_DIR}/src/ast/LiteralString.cpp
        ${ast_optimizer_SOURCE_DIR}/src/ast/LiteralFloat.cpp
        ${ast_optimizer_SOURCE_DIR}/src/ast/VarDecl.cpp
        ${ast_optimizer_SOURCE_DIR}/src/ast/Block.cpp
        ${ast_optimizer_SOURCE_DIR}/src/ast/Operator.cpp
        ${ast_optimizer_SOURCE_DIR}/src/ast/BinaryExpr.cpp
        ${ast_optimizer_SOURCE_DIR}/src/ast/VarAssignm.cpp
        ${ast_optimizer_SOURCE_DIR}/src/ast/Group.cpp
        ${ast_optimizer_SOURCE_DIR}/src/ast/If.cpp
        ${ast_optimizer_SOURCE_DIR}/src/ast/Call.cpp
        ${ast_optimizer_SOURCE_DIR}/src/ast/CallExternal.cpp
        ${ast_optimizer_SOURCE_DIR}/src/ast/LogicalExpr.cpp
        ${ast_optimizer_SOURCE_DIR}/src/ast/Return.cpp
        ${ast_optimizer_SOURCE_DIR}/src/ast/UnaryExpr.cpp
        ${ast_optimizer_SOURCE_DIR}/src/ast/While.cpp
        ${ast_optimizer_SOURCE_DIR}/src/ast/Function.cpp
        ${ast_optimizer_SOURCE_DIR}/src/ast/FunctionParameter.cpp
        ${ast_optimizer_SOURCE_DIR}/src/ast/CallExternal.cpp
        ${ast_optimizer_SOURCE_DIR}/src/ast/Ast.cpp
        ${ast_optimizer_SOURCE_DIR}/src/visitor/Visitor.cpp
        ${ast_optimizer_SOURCE_DIR}/src/visitor/PrintVisitor.cpp
        ${ast_optimizer_SOURCE_DIR}/src/visitor/MultRewriteVisitor.cpp
        ${ast_optimizer_SOURCE_DIR}/src/ast/Node.cpp
        ${ast_optimizer_SOURCE_DIR}/src/utilities/Scope.cpp
        ${ast_optimizer_SOURCE_DIR}/examples/genAstDemo.cpp
        ${ast_optimizer_SOURCE_DIR}/include/utilities/RandNumGen.h
        ${ast_optimizer_SOURCE_DIR}/include/utilities/Datatypes.h
        ${ast_optimizer_SOURCE_DIR}/src/visitor/MultDepthVisitor.cpp
        ${ast_optimizer_SOURCE_DIR}/include/visitor/MultDepthVisitor.h
        ${ast_optimizer_SOURCE_DIR}/src/optimizer/ConeRewriter.cpp)

add_executable(ast_demo
        ${ast_optimizer_SOURCE_DIR}/main.cpp
        ${SOURCE_FILES})

target_link_libraries(ast_demo PRIVATE nlohmann_json::nlohmann_json)

##############################
# TARGET: ast_lib
# built by subdirectory test/CMakeLists.txt
##############################

# build a library out of the code to be used by the tests
add_library(ast_lib ${SOURCE_FILES})
target_link_libraries(ast_lib PUBLIC nlohmann_json::nlohmann_json)

add_subdirectory(test)

