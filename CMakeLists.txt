cmake_minimum_required(VERSION 3.14)
project(ast_optimizer
        VERSION 0.1
        DESCRIPTION "Optimizer for Fully Homomorphic Encryption (FHE) working on an AST-like (Abstract Syntax Tree) representation of a program"
        LANGUAGES CXX)

##############################
# Download and install nlohmann-json if required
##############################

find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
    message("Downloading nlohmann_json")
    include(FetchContent)
    set(JSON_BuildTests OFF CACHE INTERNAL "")
    FetchContent_Declare(
            nlohmann_json
            GIT_REPOSITORY    https://github.com/nlohmann/json.git
            GIT_TAG           v3.7.3)
    FetchContent_MakeAvailable(nlohmann_json)
endif()

##############################
# TARGET: ast_opt
#
# The main ast_optimizer as a library
##############################

set(SOURCE_FILES
        ${ast_optimizer_SOURCE_DIR}/src/ast/AbstractExpr.cpp
        ${ast_optimizer_SOURCE_DIR}/src/ast/AbstractLiteral.cpp
        ${ast_optimizer_SOURCE_DIR}/src/ast/AbstractNode.cpp
        ${ast_optimizer_SOURCE_DIR}/src/ast/AbstractStatement.cpp
        ${ast_optimizer_SOURCE_DIR}/src/ast/Ast.cpp
        ${ast_optimizer_SOURCE_DIR}/src/ast/ArithmeticExpr.cpp
        ${ast_optimizer_SOURCE_DIR}/src/ast/OperatorExpr.cpp
        ${ast_optimizer_SOURCE_DIR}/src/ast/Block.cpp
        ${ast_optimizer_SOURCE_DIR}/src/ast/Call.cpp
        ${ast_optimizer_SOURCE_DIR}/src/ast/CallExternal.cpp
        ${ast_optimizer_SOURCE_DIR}/src/ast/Datatype.cpp
        ${ast_optimizer_SOURCE_DIR}/src/ast/Function.cpp
        ${ast_optimizer_SOURCE_DIR}/src/ast/FunctionParameter.cpp
        ${ast_optimizer_SOURCE_DIR}/src/ast/If.cpp
        ${ast_optimizer_SOURCE_DIR}/src/ast/LiteralBool.cpp
        ${ast_optimizer_SOURCE_DIR}/src/ast/LiteralFloat.cpp
        ${ast_optimizer_SOURCE_DIR}/src/ast/LiteralInt.cpp
        ${ast_optimizer_SOURCE_DIR}/src/ast/LiteralString.cpp
        ${ast_optimizer_SOURCE_DIR}/src/ast/LogicalExpr.cpp
        ${ast_optimizer_SOURCE_DIR}/src/ast/Operator.cpp
        ${ast_optimizer_SOURCE_DIR}/src/ast/ParameterList.cpp
        ${ast_optimizer_SOURCE_DIR}/src/ast/Return.cpp
        ${ast_optimizer_SOURCE_DIR}/src/ast/UnaryExpr.cpp
        ${ast_optimizer_SOURCE_DIR}/src/ast/VarAssignm.cpp
        ${ast_optimizer_SOURCE_DIR}/src/ast/VarDecl.cpp
        ${ast_optimizer_SOURCE_DIR}/src/ast/Variable.cpp
        ${ast_optimizer_SOURCE_DIR}/src/ast/While.cpp
        ${ast_optimizer_SOURCE_DIR}/src/optimizer/ConeRewriter.cpp
        ${ast_optimizer_SOURCE_DIR}/src/utilities/MultiplicativeDepthCalculator.cpp
        ${ast_optimizer_SOURCE_DIR}/src/utilities/NodeUtils.cpp
        ${ast_optimizer_SOURCE_DIR}/src/utilities/Scope.cpp
        ${ast_optimizer_SOURCE_DIR}/src/visitor/EvaluationVisitor.cpp
        ${ast_optimizer_SOURCE_DIR}/src/visitor/MultDepthVisitor.cpp
        ${ast_optimizer_SOURCE_DIR}/src/visitor/MultRewriteVisitor.cpp
        ${ast_optimizer_SOURCE_DIR}/src/visitor/PrintVisitor.cpp
        ${ast_optimizer_SOURCE_DIR}/src/visitor/SecretTaintingVisitor.cpp
        ${ast_optimizer_SOURCE_DIR}/src/visitor/Visitor.cpp
        ${ast_optimizer_SOURCE_DIR}/test/AstTestingGenerator.cpp
        ${ast_optimizer_SOURCE_DIR}/src/utilities/EvaluationPrinter.cpp
        ${ast_optimizer_SOURCE_DIR}/src/utilities/DotPrinter.cpp
        ${ast_optimizer_SOURCE_DIR}/src/visitor/CompileTimeExpressionSimplifier.cpp
        ${ast_optimizer_SOURCE_DIR}/src/ast/AbstractBinaryExpr.cpp
        ${ast_optimizer_SOURCE_DIR}/src/ast/For.cpp
        ${ast_optimizer_SOURCE_DIR}/src/ast/Rotate.cpp
        ${ast_optimizer_SOURCE_DIR}/src/ast/AbstractMatrix.cpp
        ${ast_optimizer_SOURCE_DIR}/src/ast/Dimension.cpp
        ${ast_optimizer_SOURCE_DIR}/src/ast/Matrix.cpp
        ${ast_optimizer_SOURCE_DIR}/src/ast/Transpose.cpp
        ${ast_optimizer_SOURCE_DIR}/src/ast/MatrixElementRef.cpp
        ${ast_optimizer_SOURCE_DIR}/src/visitor/ControlFlowGraphVisitor.cpp
        ${ast_optimizer_SOURCE_DIR}/src/utilities/GraphNode.cpp
        ${ast_optimizer_SOURCE_DIR}/src/utilities/NodeRelationship.cpp
        ${ast_optimizer_SOURCE_DIR}/src/evaluation/EvaluationAlgorithms.cpp
        ${ast_optimizer_SOURCE_DIR}/src/ast/MatrixAssignm.cpp
        ${ast_optimizer_SOURCE_DIR}/src/ast/GetMatrixSize.cpp
        ${ast_optimizer_SOURCE_DIR}/src/optimizer/BatchingChecker.cpp)

set(HEADER_ONLY_SOURCES
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/ast/OpSymbEnum.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/utilities/RandNumGen.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/utilities/TestUtils.h)

# TODO (aviand): Go through this list and determine which headers aren't actually public
set(PUBLIC_HEADERS
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/ast/AbstractBinaryExpr.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/ast/AbstractExpr.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/ast/AbstractLiteral.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/ast/AbstractMatrix.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/ast/AbstractNode.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/ast/AbstractStatement.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/ast/ArithmeticExpr.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/ast/Ast.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/ast/Block.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/ast/Call.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/ast/CallExternal.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/ast/Datatype.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/ast/Dimension.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/ast/For.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/ast/Function.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/ast/FunctionParameter.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/ast/GetMatrixSize.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/ast/If.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/ast/LiteralBool.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/ast/LiteralFloat.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/ast/LiteralInt.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/ast/LiteralString.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/ast/LogicalExpr.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/ast/Matrix.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/ast/MatrixAssignm.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/ast/MatrixElementRef.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/ast/Operator.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/ast/OperatorExpr.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/ast/OpSymbEnum.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/ast/ParameterList.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/ast/Return.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/ast/Rotate.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/ast/Transpose.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/ast/UnaryExpr.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/ast/VarAssignm.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/ast/VarDecl.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/ast/Variable.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/ast/While.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/evaluation/EvaluationAlgorithms.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/optimizer/ConeRewriter.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/utilities/DotPrinter.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/utilities/EvaluationPrinter.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/utilities/GraphNode.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/utilities/MultiplicativeDepthCalculator.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/utilities/NodeRelationship.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/utilities/NodeUtils.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/utilities/RandNumGen.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/utilities/Scope.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/utilities/TestUtils.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/visitor/CompileTimeExpressionSimplifier.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/visitor/ControlFlowGraphVisitor.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/visitor/EvaluationVisitor.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/visitor/MultDepthVisitor.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/visitor/MultRewriteVisitor.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/visitor/PrintVisitor.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/visitor/SecretTaintingVisitor.h
        ${ast_optimizer_SOURCE_DIR}/include/ast_opt/visitor/Visitor.h)


# build a library out of the code to be used by the tests
add_library(ast_opt ${SOURCE_FILES} ${HEADER_ONLY_SOURCES})
add_library(ast_opt::ast_opt ALIAS ast_opt)
target_link_libraries(ast_opt PUBLIC nlohmann_json::nlohmann_json)
target_include_directories(ast_opt PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
set_target_properties(ast_opt PROPERTIES PUBLIC_HEADER ${PUBLIC_HEADERS})
target_compile_features(ast_opt PUBLIC cxx_std_17)

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
        EXPORT  ast_optTargets
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
        ${ast_optimizer_SOURCE_DIR}/examples/main.cpp
        ${ast_optimizer_SOURCE_DIR}/examples/genAstDemo.cpp)

target_link_libraries(ast_demo PRIVATE ast_opt)

####################################################################################
# TESTING
#
# only if this is the root project
####################################################################################
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    message("Enabling ast_opt Tests")
    enable_testing()
    add_subdirectory(test EXCLUDE_FROM_ALL)
endif()
