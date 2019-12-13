cmake_minimum_required(VERSION 3.15)
set(CMAKE_CXX_STANDARD 17)
project(master_thesis_code)

##############################
# Download nlohmann-json
##############################

# exclude the tests from the build
set(JSON_BuildTests OFF CACHE INTERNAL "")

configure_file(libs/nlohmann-json.txt.in nlohmann-json-download/CMakeLists.txt)
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


# Add googletest directly to our build. This defines the gtest and gtest_main targets.
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/nlohmann-json-src
        ${CMAKE_CURRENT_BINARY_DIR}/nlohmann-json-build
        EXCLUDE_FROM_ALL)

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
        ${master_thesis_code_SOURCE_DIR}/src/ast/CallExternal.cpp
        ${master_thesis_code_SOURCE_DIR}/src/ast/Ast.cpp
        ${master_thesis_code_SOURCE_DIR}/src/visitor/Visitor.cpp
        ${master_thesis_code_SOURCE_DIR}/src/visitor/StatisticsVisitor.cpp
        )

add_executable(master_thesis_code
        ${master_thesis_code_SOURCE_DIR}/main.cpp
        ${SOURCE_FILES})

target_link_libraries(master_thesis_code PRIVATE nlohmann_json::nlohmann_json)


##############################
# TARGET: master_thesis_code_lib
# built by subdirectory test/CMakeLists.txt
##############################

add_subdirectory(test)

# build a library out of the code to be used by
add_library(master_thesis_code_lib ${SOURCE_FILES})
target_link_libraries(master_thesis_code_lib PRIVATE nlohmann_json::nlohmann_json)



