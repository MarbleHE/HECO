cmake_minimum_required(VERSION 3.11.0)
include(FetchContent) # Introduced in CMake 3.11
include(GoogleTest) # Introduced in CMake 3.10

include_directories("${PROJECT_SOURCE_DIR}")

##############################
# Download GoogleTest framework
##############################
FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.10.0
)
FetchContent_MakeAvailable(googletest)



##############################
# TARGET: testing
##############################
set(TEST_FILES
        AstTestingGenerator.cpp
        AstTestingGenerator.h
        ast/BlockTest.cpp
        ast/IfTest.cpp
        ast/LiteralTest.cpp
        ast/VariableTest.cpp
        ast/VariableAssignmentTest.cpp
        ast/VariableDeclarationTest.cpp
        JsonOutputTest.cpp
        visitor/PrintVisitorTest.cpp
        visitor/VisitorTest.cpp
        )

add_executable(testing-all
        ${TEST_FILES})

# this is important to have code coverage in CLion
if (CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "--coverage")
elseif ("${CMAKE_C_COMPILER_ID}" MATCHES "(Apple)?[Cc]lang" OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "(Apple)?[Cc]lang")
    set(CMAKE_CXX_FLAGS "-fprofile-instr-generate -fcoverage-mapping")
endif ()

target_link_libraries(testing-all PRIVATE gtest ast_opt gtest_main)
target_include_directories(testing-all PRIVATE ${ast_optimizer_SOURCE_DIR}/include/)

if(MSVC)
# Mark gtest as external
target_compile_options(testing-all PRIVATE /external:I${gtest_SOURCE_DIR}/include)
endif()

# create ctest targets
gtest_discover_tests(testing-all TEST_PREFIX gtest:)

# make sure that ast_lib is built before the testing-all target
add_dependencies(testing-all ast_opt)
