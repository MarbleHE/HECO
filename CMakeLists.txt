cmake_minimum_required(VERSION 3.15)
project(master_thesis_code)

set(CMAKE_CXX_STANDARD 14)

include_directories(include/src)
#add_executable(master_thesis_code main.cpp src/ast/excluded/ExpressionStmt.cpp include/ast/excluded/ExpressionStmt.h src/ast/excluded/BinaryExpr.cpp include/ast/excluded/BinaryExpr.h src/ast/excluded/Operator.cpp include/ast/excluded/Operator.h src/ast/excluded/While.cpp include/ast/excluded/While.h src/ast/excluded/AbstractStatement.cpp include/ast/excluded/AbstractStatement.h src/ast/excluded/VarDecl.cpp include/ast/excluded/VarDecl.h src/ast/excluded/Return.cpp include/ast/excluded/Return.h src/ast/excluded/If.cpp include/ast/excluded/If.h src/ast/excluded/Function.cpp include/ast/excluded/Function.h src/ast/excluded/Class.cpp include/ast/excluded/Class.h src/ast/excluded/Block.cpp include/ast/excluded/Block.h src/ast/excluded/Variable.cpp include/ast/excluded/Variable.h src/ast/excluded/UnaryExpr.cpp include/ast/excluded/UnaryExpr.h src/ast/excluded/LogicalExpr.cpp include/ast/excluded/LogicalExpr.h src/ast/excluded/Literal.cpp include/ast/excluded/Literal.h src/ast/excluded/LiteralInt.cpp include/ast/excluded/LiteralInt.h src/ast/excluded/LiteralString.cpp include/ast/excluded/LiteralString.h src/ast/excluded/LiteralBool.cpp include/ast/excluded/LiteralBool.h src/ast/excluded/Group.cpp include/ast/excluded/Group.h src/ast/excluded/Call.cpp include/ast/excluded/Call.h src/ast/excluded/VarAssignm.cpp include/ast/excluded/VarAssignm.h src/ast/AbstractExpr.cpp include/ast/AbstractExpr.h FunctionParameter.cpp FunctionParameter.h)

add_executable(master_thesis_code
        main.cpp
        include/ast/AbstractExpr.h
        src/ast/AbstractExpr.cpp
        include/ast/AbstractStatement.h
        src/ast/AbstractStatement.cpp
        include/ast/Variable.h
        src/ast/Variable.cpp
        include/ast/Literal.h
        src/ast/Literal.cpp
        include/ast/LiteralBool.h
        src/ast/LiteralBool.cpp
        include/ast/LiteralInt.h
        src/ast/LiteralInt.cpp
        include/ast/LiteralString.h
        src/ast/LiteralString.cpp
        include/ast/VarDecl.h
        src/ast/VarDecl.cpp
        include/ast/Block.h
        src/ast/Block.cpp
        include/ast/Operator.h
        src/ast/Operator.cpp
        include/ast/BinaryExpr.h
        src/ast/BinaryExpr.cpp
        include/ast/VarAssignm.h
        src/ast/VarAssignm.cpp
        include/ast/Group.h
        src/ast/Group.cpp
        include/ast/If.h
        src/ast/If.cpp
        include/ast/Call.h
        src/ast/Call.cpp
        include/ast/Class.h
        src/ast/Class.cpp
        include/ast/ExpressionStmt.h
        src/ast/ExpressionStmt.cpp
        include/ast/LogicalExpr.h
        src/ast/LogicalExpr.cpp
        include/ast/Return.h
        src/ast/Return.cpp
        include/ast/UnaryExpr.h
        src/ast/UnaryExpr.cpp
        include/ast/While.h
        src/ast/While.cpp
        include/ast/Function.h
        src/ast/Function.cpp
        include/ast/FunctionParameter.h
        src/ast/FunctionParameter.cpp
        main.h)