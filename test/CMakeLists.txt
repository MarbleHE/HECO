include_directories("${PROJECT_SOURCE_DIR}")


##############################
# Download GoogleTest framework
##############################

# Download and unpack googletest at configure time
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

# Add googletest directly to our build. This defines the gtest and gtest_main targets.
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
        ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
        EXCLUDE_FROM_ALL)

# The gtest/gtest_main targets carry header search path dependencies automatically when using CMake 2.8.11 or later.
# Otherwise we have to add them here ourselves.
if (CMAKE_VERSION VERSION_LESS 2.8.11)
    include_directories("${gtest_SOURCE_DIR}/include")
endif ()

##############################
# TARGET: testing
##############################

set(TEST_FILES
        BinaryExprTest.cpp
        JsonOutputTest.cpp
        PrintVisitorTest.cpp
        MultRewriteTest.cpp
        AstTestingGenerator.cpp
        AstTestingGenerator.h AstEvaluationTest.cpp "../include/utilities/TestUtils .h")

add_executable(testing-all
        main.cpp
        ${TEST_FILES})

# this is important to have code coverage in CLion
set(CMAKE_CXX_FLAGS "-fprofile-instr-generate -fcoverage-mapping")

target_link_libraries(testing-all gtest master_thesis_code_lib nlohmann_json::nlohmann_json)

# make sure that master_thesis_code_lib is built before the testing-all target
add_dependencies(testing-all master_thesis_code_lib)

