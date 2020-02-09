include_directories("${PROJECT_SOURCE_DIR}")

##############################
# TARGET: testing
##############################

set(TEST_FILES
        BinaryExprTest.cpp
        JsonOutputTest.cpp
        PrintVisitorTest.cpp
        MultRewriteTest.cpp
        AstTestingGenerator.cpp
        AstTestingGenerator.h AstEvaluationTest.cpp
        ${master_thesis_code_SOURCE_DIR}/include/utilities/TestUtils.h
        MultDepthVisitorTest.cpp
        ChildParentTest.cpp
        AstClassTest.cpp
        DotPrinterTest.cpp)

add_executable(testing-all
        main.cpp
        ${TEST_FILES})

# this is important to have code coverage in CLion
if (CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "--coverage")
elseif ("${CMAKE_C_COMPILER_ID}" MATCHES "(Apple)?[Cc]lang" OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "(Apple)?[Cc]lang")
    set(CMAKE_CXX_FLAGS "-fprofile-instr-generate -fcoverage-mapping")
endif ()

target_link_libraries(testing-all gtest ast_lib nlohmann_json::nlohmann_json)

# make sure that master_thesis_code_lib is built before the testing-all target
add_dependencies(testing-all ast_lib)

