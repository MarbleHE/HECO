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
        MultDepthVisitorTest.cpp)

add_executable(testing-all
        main.cpp
        ${TEST_FILES})

# this is important to have code coverage in CLion
set(CMAKE_CXX_FLAGS "-fprofile-instr-generate -fcoverage-mapping")

target_link_libraries(testing-all gtest master_thesis_code_lib nlohmann_json::nlohmann_json)

# make sure that master_thesis_code_lib is built before the testing-all target
add_dependencies(testing-all master_thesis_code_lib)

