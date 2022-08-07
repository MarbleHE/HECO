cmake_minimum_required(VERSION 3.19.0)
include(FetchContent) # Introduced in CMake 3.11
include(GoogleTest) # Introduced in CMake 3.10

include_directories("${PROJECT_SOURCE_DIR}")

##############################
# Download GoogleTest framework
##############################
if(NOT TARGET gtest)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.12.1
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    get_property(
        compile_options
        DIRECTORY
        PROPERTY COMPILE_OPTIONS
    )
    
    set_property(
        DIRECTORY
        APPEND
        PROPERTY COMPILE_OPTIONS -w
    )
    
    FetchContent_MakeAvailable(googletest)
    
    set_property(
        DIRECTORY
        PROPERTY COMPILE_OPTIONS ${compile_options}
    )
    
endif()



##############################
# TARGET: testing
##############################
set(TEST_FILES       
       BoxBlurTest.cpp
       CardioTest.cpp
       DotProductTest.cpp
       GxKernelTest.cpp
       GyKernelTest.cpp
       HammingDistanceTest.cpp
       L2DistanceTest.cpp
       LaplaceSharpeningTest.cpp
       LinearKernelTest.cpp
       MatrixVectorProductTest.cpp
       PolynomialKernelTest.cpp
       RobertsCrossTest.cpp
)


add_executable(testing-all
        ${TEST_FILES})

# this is important to have code coverage in CLion
#if (CMAKE_COMPILER_IS_GNUCXX)
#    set(CMAKE_CXX_FLAGS "--coverage")
#elseif ("${CMAKE_C_COMPILER_ID}" MATCHES "(Apple)?[Cc]lang" OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "(Apple)?[Cc]lang")
#    set(CMAKE_CXX_FLAGS "-fprofile-instr-generate -fcoverage-mapping")
#endif ()

target_link_libraries(testing-all PRIVATE gtest gtest_main)
target_include_directories(testing-all PRIVATE ${abc_SOURCE_DIR}/include/ bench)

if(MSVC)
    # Mark gtest as external
    target_compile_options(testing-all PRIVATE /external:I${gtest_SOURCE_DIR}/include)
endif()

# create ctest targets
gtest_discover_tests(testing-all TEST_PREFIX gtest:)
add_dependencies(testing-all mlir-headers)
