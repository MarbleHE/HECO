cmake_minimum_required(VERSION 3.11.0)
include(FetchContent) # Introduced in CMake 3.11
include(GoogleTest) # Introduced in CMake 3.10

include_directories("${PROJECT_SOURCE_DIR}")

##############################
# Download GoogleTest framework
##############################
if(NOT AST_OPTIMIZER_SKIP_GTEST_INSTALL)
    FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG release-1.10.0
    )

    FetchContent_GetProperties(googletest)
    if (NOT googletest_POPULATED)
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        FetchContent_Populate(googletest)
        add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})
    endif ()
endif()
##############################
# TARGET: testing
##############################

include_directories(
        ${ast_optimizer_SOURCE_DIR}/include/ast
        ${ast_optimizer_SOURCE_DIR}/include/optimizer
        ${ast_optimizer_SOURCE_DIR}/include/utilities
        ${ast_optimizer_SOURCE_DIR}/include/visitor
        ${ast_optimizer_SOURCE_DIR}/examples
        ${ast_optimizer_SOURCE_DIR}/test)

set(TEST_FILES
        ArithmeticExprTest.cpp
        JsonOutputTest.cpp
        PrintVisitorTest.cpp
        MultRewriteTest.cpp
        AstTestingGenerator.cpp
        AstTestingGenerator.h
        EvaluationVisitorTest.cpp
        ${ast_optimizer_SOURCE_DIR}/include/utilities/TestUtils.h
        MultDepthVisitorTest.cpp
        ChildParentTest.cpp
        AstClassTest.cpp
        DotPrinterTest.cpp
        ConeRewriterTest.cpp
        LiteralTest.cpp
        NodeTest.cpp
        NodeCloneRecursiveDeepTest.cpp
        CompileTimeExpressionSimplifierTest.cpp
        SecretTaintingTest.cpp
        NodeUtilsTest.cpp
        MatrixTest.cpp)

add_executable(testing-all
        ${TEST_FILES})

# this is important to have code coverage in CLion
if (CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "--coverage")
elseif ("${CMAKE_C_COMPILER_ID}" MATCHES "(Apple)?[Cc]lang" OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "(Apple)?[Cc]lang")
    set(CMAKE_CXX_FLAGS "-fprofile-instr-generate -fcoverage-mapping")
endif ()

target_link_libraries(testing-all gtest ast_lib nlohmann_json::nlohmann_json gtest_main)

# create ctest targets
gtest_discover_tests(testing-all TEST_PREFIX gtest:)

# make sure that ast_lib is built before the testing-all target
add_dependencies(testing-all ast_lib)
