cmake_minimum_required (VERSION 3.9...3.17)

# For older versions that do not automatically use highest allowed policy
if(${CMAKE_VERSION} VERSION_LESS 3.12)
    cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()

# Required for selecting MSVC runtime library
if (MSVC)
    cmake_policy(SET CMP0091 NEW)
endif()

project(ast_optimizer
        VERSION 0.1
        DESCRIPTION "Optimizer for Fully Homomorphic Encryption (FHE) working on an AST-like (Abstract Syntax Tree) representation of a program"
        LANGUAGES CXX)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/Modules" ${CMAKE_MODULE_PATH})

set(CXXFLAGS "--coverage")

if (MSVC)
    message("Using MSVC")

    # Force static MSVC runtime library on Windows
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

    # Switch MSVC into multi threaded mode
    add_compile_options(/MP)

    # Force MSVC to use slightly less broken template rules
    add_compile_options(/permissive-)
endif ()

##############################
# Determine if SEAL is available
##############################

find_package(SEAL) # sets SEAL_FOUND variable

##############################
# Download and install nlohmann-json if required
##############################

find_package(nlohmann_json QUIET)
if (NOT nlohmann_json_FOUND)
    message("Downloading nlohmann_json")
    include(FetchContent)
    set(JSON_BuildTests OFF CACHE INTERNAL "")
    FetchContent_Declare(
            nlohmann_json
            GIT_REPOSITORY https://github.com/nlohmann/json.git
            GIT_TAG v3.7.3)
    FetchContent_MakeAvailable(nlohmann_json)
endif ()

##############################
# TARGET: ast_opt
#
# The main ast_optimizer as a library
##############################

set(SOURCE_FILES
        ${ast_optimizer_SOURCE_DIR}/src/ast/AbstractNode.cpp
        ${ast_optimizer_SOURCE_DIR}/src/ast/BinaryExpression.cpp
        ${ast_optimizer_SOURCE_DIR}/src/ast/Block.cpp
        ${ast_optimizer_SOURCE_DIR}/src/ast/For.cpp
        ${ast_optimizer_SOURCE_DIR}/src/ast/If.cpp
        ${ast_optimizer_SOURCE_DIR}/src/ast/UnaryExpression.cpp
        ${ast_optimizer_SOURCE_DIR}/src/ast/Variable.cpp
        ${ast_optimizer_SOURCE_DIR}/src/ast/VariableAssignment.cpp
        ${ast_optimizer_SOURCE_DIR}/src/ast/VariableDeclaration.cpp
        ${ast_optimizer_SOURCE_DIR}/src/utilities/Datatype.cpp
        ${ast_optimizer_SOURCE_DIR}/src/utilities/Operator.cpp
        ${ast_optimizer_SOURCE_DIR}/src/visitor/PrintVisitor.cpp
        ${ast_optimizer_SOURCE_DIR}/src/visitor/Visitor.cpp)

set(HEADER_ONLY_SOURCES
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/ast/AbstractExpression.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/ast/AbstractStatement.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/ast/AbstractTarget.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/visitor/IVisitor.h)

# TODO: Go through this list and determine which headers aren't actually public
set(PUBLIC_HEADERS
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/ast/AbstractExpression.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/ast/AbstractNode.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/ast/AbstractStatement.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/ast/Block.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/ast/For.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/ast/If.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/ast/Literal.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/ast/Variable.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/ast/VariableAssignment.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/ast/VariableDeclaration.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/utilities/Datatype.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/utilities/Operator.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/visitor/IVisitor.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/visitor/PrintVisitor.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/visitor/Visitor.h)


# build a library out of the code to be used by the tests
add_library(ast_opt ${SOURCE_FILES} ${HEADER_ONLY_SOURCES})
add_library(ast_opt::ast_opt ALIAS ast_opt)
target_link_libraries(ast_opt PUBLIC nlohmann_json::nlohmann_json)
target_include_directories(ast_opt PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
set_target_properties(ast_opt PROPERTIES PUBLIC_HEADER "${PUBLIC_HEADERS}")
target_compile_features(ast_opt PUBLIC cxx_std_17)

if (SEAL_FOUND)
    message("Enabling SEAL functionality")
    target_include_directories(ast_opt PUBLIC ${SEAL_INCLUDE_DIRS})
    target_link_libraries(ast_opt PUBLIC ${SEAL_LIBRARIES})
    target_compile_definitions(ast_opt PUBLIC -DHAVE_SEAL_BFV)
    target_compile_definitions(ast_opt PUBLIC -DHAVE_SEAL_CKKS)
endif(SEAL_FOUND)

####################################################################################
## ast_opt installation
##
## ast_optimizer installed globally as a library
####################################################################################
# Create a Version File
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
        ast_optConfigVersion.cmake
        VERSION ${PACKAGE_VERSION}
        COMPATIBILITY AnyNewerVersion
)

# Configure (i.e. generate) the ast_optConfig.cmake file and copy both it and the version to install
configure_file(ast_optConfig.cmake.in ast_optConfig.cmake @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/ast_optConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/ast_optConfigVersion.cmake"
        DESTINATION lib/cmake/ast_opt)

install(TARGETS ast_opt
        EXPORT ast_optTargets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        PUBLIC_HEADER DESTINATION include
        INCLUDES DESTINATION include/ast_opt)

install(DIRECTORY "include/"
        DESTINATION "include")

install(EXPORT ast_optTargets
        FILE ast_optTargets.cmake
        NAMESPACE ast_opt::
        DESTINATION lib/cmake/ast_opt)


##############################
# TARGET: ast_demo
#
# A simple example of the ast_optimizer
##############################

add_executable(ast_demo
        ${ast_optimizer_SOURCE_DIR}/examples/main.cpp)

target_link_libraries(ast_demo PRIVATE ast_opt)

####################################################################################
# TESTING
#
# only if this is the root project
####################################################################################
if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    message("Enabling ast_opt Tests")
    enable_testing()
    add_subdirectory(test EXCLUDE_FROM_ALL)
endif ()
