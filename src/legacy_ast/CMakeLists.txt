##############################
# TARGET: cpp_ast
#
# Functionality around the legacy C++ ast
##############################

# All source files (*.cpp) for ABC
set(SOURCE_FILES
        ${CMAKE_CURRENT_LIST_DIR}/ast/AbstractNode.cpp
        ${CMAKE_CURRENT_LIST_DIR}/ast/Assignment.cpp
        ${CMAKE_CURRENT_LIST_DIR}/ast/BinaryExpression.cpp
        ${CMAKE_CURRENT_LIST_DIR}/ast/Block.cpp
        ${CMAKE_CURRENT_LIST_DIR}/ast/Call.cpp
        ${CMAKE_CURRENT_LIST_DIR}/ast/ExpressionList.cpp
        ${CMAKE_CURRENT_LIST_DIR}/ast/For.cpp
        ${CMAKE_CURRENT_LIST_DIR}/ast/Function.cpp
        ${CMAKE_CURRENT_LIST_DIR}/ast/FunctionParameter.cpp
        ${CMAKE_CURRENT_LIST_DIR}/ast/If.cpp
        ${CMAKE_CURRENT_LIST_DIR}/ast/IndexAccess.cpp
        ${CMAKE_CURRENT_LIST_DIR}/ast/OperatorExpression.cpp
        ${CMAKE_CURRENT_LIST_DIR}/ast/Return.cpp
        ${CMAKE_CURRENT_LIST_DIR}/ast/TernaryOperator.cpp
        ${CMAKE_CURRENT_LIST_DIR}/ast/UnaryExpression.cpp
        ${CMAKE_CURRENT_LIST_DIR}/ast/Variable.cpp
        ${CMAKE_CURRENT_LIST_DIR}/ast/VariableDeclaration.cpp
        ${CMAKE_CURRENT_LIST_DIR}/ast_parser/Errors.cpp
        ${CMAKE_CURRENT_LIST_DIR}/ast_parser/File.cpp
        ${CMAKE_CURRENT_LIST_DIR}/ast_utilities/Datatype.cpp
        ${CMAKE_CURRENT_LIST_DIR}/ast_utilities/NodeUtils.cpp
        ${CMAKE_CURRENT_LIST_DIR}/ast_utilities/Operator.cpp
        ${CMAKE_CURRENT_LIST_DIR}/ast_utilities/ParentSettingVisitor.cpp
        ${CMAKE_CURRENT_LIST_DIR}/ast_utilities/PlainVisitor.cpp
        ${CMAKE_CURRENT_LIST_DIR}/ast_utilities/PrintVisitor.cpp
        ${CMAKE_CURRENT_LIST_DIR}/ast_utilities/ProgramPrintVisitor.cpp
        ${CMAKE_CURRENT_LIST_DIR}/ast_utilities/AbcAstToMlirVisitor.cpp
        ${CMAKE_CURRENT_LIST_DIR}/ast_utilities/Scope.cpp
        ${CMAKE_CURRENT_LIST_DIR}/ast_utilities/ScopedVisitor.cpp
        ${CMAKE_CURRENT_LIST_DIR}/ast_parser/Parser.cpp
        ${CMAKE_CURRENT_LIST_DIR}/ast_parser/PushBackStream.cpp
        ${CMAKE_CURRENT_LIST_DIR}/ast_parser/Tokenizer.cpp
        ${CMAKE_CURRENT_LIST_DIR}/ast_parser/Tokens.cpp
        )

# This contains all the header-only files (e.g., abstract classes, interfaces and templates that don't have *.cpp file)
set(HEADER_ONLY_SOURCES
        ${CMAKE_SOURCE_DIR}/include/heco/legacy_ast/ast/AbstractExpression.h
        ${CMAKE_SOURCE_DIR}/include/heco/legacy_ast/ast/AbstractStatement.h
        ${CMAKE_SOURCE_DIR}/include/heco/legacy_ast/ast/AbstractTarget.h
        ${CMAKE_SOURCE_DIR}/include/heco/legacy_ast/ast/Ast.h
        ${CMAKE_SOURCE_DIR}/include/heco/legacy_ast/ast_utilities/IVisitor.h
        ${CMAKE_SOURCE_DIR}/include/heco/legacy_ast/ast_utilities/VariableMap.h
        ${CMAKE_SOURCE_DIR}/include/heco/legacy_ast/ast_utilities/Visitor.h)

# This contains the headers that developers that want to link directly against ABC might need to have access to
# These are the headers that are installed when ABC is installed as a system-wide library
set(PUBLIC_HEADERS
        ${CMAKE_SOURCE_DIR}/include/heco/legacy_ast/ast/AbstractExpression.h
        ${CMAKE_SOURCE_DIR}/include/heco/legacy_ast/ast/AbstractNode.h
        ${CMAKE_SOURCE_DIR}/include/heco/legacy_ast/ast/AbstractStatement.h
        ${CMAKE_SOURCE_DIR}/include/heco/legacy_ast/ast/Assignment.h
        ${CMAKE_SOURCE_DIR}/include/heco/legacy_ast/ast/BinaryExpression.h
        ${CMAKE_SOURCE_DIR}/include/heco/legacy_ast/ast/Block.h
        ${CMAKE_SOURCE_DIR}/include/heco/legacy_ast/ast/Call.h
        ${CMAKE_SOURCE_DIR}/include/heco/legacy_ast/ast/ExpressionList.h
        ${CMAKE_SOURCE_DIR}/include/heco/legacy_ast/ast/For.h
        ${CMAKE_SOURCE_DIR}/include/heco/legacy_ast/ast/Function.h
        ${CMAKE_SOURCE_DIR}/include/heco/legacy_ast/ast/FunctionParameter.h
        ${CMAKE_SOURCE_DIR}/include/heco/legacy_ast/ast/If.h
        ${CMAKE_SOURCE_DIR}/include/heco/legacy_ast/ast/IndexAccess.h
        ${CMAKE_SOURCE_DIR}/include/heco/legacy_ast/ast/Literal.h
        ${CMAKE_SOURCE_DIR}/include/heco/legacy_ast/ast/OperatorExpression.h
        ${CMAKE_SOURCE_DIR}/include/heco/legacy_ast/ast/Return.h
        ${CMAKE_SOURCE_DIR}/include/heco/legacy_ast/ast/TernaryOperator.h
        ${CMAKE_SOURCE_DIR}/include/heco/legacy_ast/ast/UnaryExpression.h
        ${CMAKE_SOURCE_DIR}/include/heco/legacy_ast/ast/Variable.h
        ${CMAKE_SOURCE_DIR}/include/heco/legacy_ast/ast/VariableDeclaration.h
        ${CMAKE_SOURCE_DIR}/include/heco/legacy_ast/ast_parser/Errors.h
        ${CMAKE_SOURCE_DIR}/include/heco/legacy_ast/ast_parser/Parser.h
        ${CMAKE_SOURCE_DIR}/include/heco/legacy_ast/ast_parser/PushBackStream.h
        ${CMAKE_SOURCE_DIR}/include/heco/legacy_ast/ast_parser/Tokenizer.h
        ${CMAKE_SOURCE_DIR}/include/heco/legacy_ast/ast_parser/Tokens.h
        ${CMAKE_SOURCE_DIR}/include/heco/legacy_ast/ast_utilities/Datatype.h
        ${CMAKE_SOURCE_DIR}/include/heco/legacy_ast/ast_utilities/IVisitor.h
        ${CMAKE_SOURCE_DIR}/include/heco/legacy_ast/ast_utilities/NodeUtils.h
        ${CMAKE_SOURCE_DIR}/include/heco/legacy_ast/ast_utilities/Operator.h
        ${CMAKE_SOURCE_DIR}/include/heco/legacy_ast/ast_utilities/ParentSettingVisitor.h
        ${CMAKE_SOURCE_DIR}/include/heco/legacy_ast/ast_utilities/PlainVisitor.h
        ${CMAKE_SOURCE_DIR}/include/heco/legacy_ast/ast_utilities/PrintVisitor.h
        ${CMAKE_SOURCE_DIR}/include/heco/legacy_ast/ast_utilities/ProgramPrintVisitor.h
        ${CMAKE_SOURCE_DIR}/include/heco/legacy_ast/ast_utilities/Scope.h
        ${CMAKE_SOURCE_DIR}/include/heco/legacy_ast/ast_utilities/ScopedVisitor.h
        ${CMAKE_SOURCE_DIR}/include/heco/legacy_ast/ast_utilities/VariableMap.h
)

# build a library out of the code to be used by the tests
add_library(cpp_ast ${SOURCE_FILES} ${HEADER_ONLY_SOURCES})
add_library(heco::cpp_ast ALIAS cpp_ast)
target_link_libraries(cpp_ast PUBLIC nlohmann_json::nlohmann_json)
target_include_directories(cpp_ast PUBLIC ${PROJECT_SOURCE_DIR}/include)
set_target_properties(cpp_ast PROPERTIES PUBLIC_HEADER "${PUBLIC_HEADERS}")
set_target_properties(cpp_ast PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
target_compile_features(cpp_ast PUBLIC cxx_std_17)

if (SEAL_FOUND)
    message("Enabling SEAL functionality")
    target_include_directories(cpp_ast PUBLIC ${SEAL_INCLUDE_DIRS})
    target_link_libraries(cpp_ast PUBLIC SEAL::seal)
    target_compile_definitions(cpp_ast PUBLIC -DHAVE_SEAL_BFV)
endif (SEAL_FOUND)