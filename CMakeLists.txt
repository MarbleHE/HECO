cmake_minimum_required(VERSION 3.15)
set(CMAKE_CXX_STANDARD 14)
project(master_thesis_code)

##############################
# TARGET: master_thesis_code
##############################

include_directories(${master_thesis_code_SOURCE_DIR}/include/ast)

set(SOURCE_FILES
        ${master_thesis_code_SOURCE_DIR}/src/ast/AbstractExpr.cpp
        ${master_thesis_code_SOURCE_DIR}/src/ast/AbstractStatement.cpp
        ${master_thesis_code_SOURCE_DIR}/src/ast/Variable.cpp
        ${master_thesis_code_SOURCE_DIR}/src/ast/Literal.cpp
        ${master_thesis_code_SOURCE_DIR}/src/ast/LiteralBool.cpp
        ${master_thesis_code_SOURCE_DIR}/src/ast/LiteralInt.cpp
        ${master_thesis_code_SOURCE_DIR}/src/ast/LiteralString.cpp
        ${master_thesis_code_SOURCE_DIR}/src/ast/VarDecl.cpp
        ${master_thesis_code_SOURCE_DIR}/src/ast/Block.cpp
        ${master_thesis_code_SOURCE_DIR}/src/ast/Operator.cpp
        ${master_thesis_code_SOURCE_DIR}/src/ast/BinaryExpr.cpp
        ${master_thesis_code_SOURCE_DIR}/src/ast/VarAssignm.cpp
        ${master_thesis_code_SOURCE_DIR}/src/ast/Group.cpp
        ${master_thesis_code_SOURCE_DIR}/src/ast/If.cpp
        ${master_thesis_code_SOURCE_DIR}/src/ast/Call.cpp
        ${master_thesis_code_SOURCE_DIR}/src/ast/CallExternal.cpp
        ${master_thesis_code_SOURCE_DIR}/src/ast/Class.cpp
        ${master_thesis_code_SOURCE_DIR}/src/ast/LogicalExpr.cpp
        ${master_thesis_code_SOURCE_DIR}/src/ast/Return.cpp
        ${master_thesis_code_SOURCE_DIR}/src/ast/UnaryExpr.cpp
        ${master_thesis_code_SOURCE_DIR}/src/ast/While.cpp
        ${master_thesis_code_SOURCE_DIR}/src/ast/Function.cpp
        ${master_thesis_code_SOURCE_DIR}/src/ast/FunctionParameter.cpp
        ${master_thesis_code_SOURCE_DIR}/src/ast/CallExternal.cpp)

add_executable(master_thesis_code
        ${master_thesis_code_SOURCE_DIR}/main.cpp
        ${SOURCE_FILES})

##############################
# TARGET: testing
# built by subdirectory CMakeLists.txt
##############################

add_subdirectory(test)

add_library(master_thesis_code_lib ${SOURCE_FILES})