
#include <iostream>
#include "../../include/visitor/Visitor.h"
#include "Ast.h"
#include <Function.h>
#include <Operator.h>
#include <If.h>
#include <LiteralInt.h>
#include <LiteralBool.h>
#include <LogicalExpr.h>
#include <VarAssignm.h>
#include <Group.h>
#include <Block.h>
#include <Return.h>
#include <BinaryExpr.h>
#include <Ast.h>
#include <CallExternal.h>
#include <While.h>
#include <UnaryExpr.h>
#include <Call.h>
#include <Class.h>

void Visitor::visit(Ast &elem) {
    elem.getRootNode()->accept(*this);
}

void Visitor::visit(BinaryExpr &elem) {
    elem.getLeft()->accept(*this);
    elem.getOp().accept(*this);
    elem.getRight()->accept(*this);
}

void Visitor::visit(Block &elem) {
    for (auto &stat : *elem.getStatements()) {
        stat->accept(*this);
    }
}

void Visitor::visit(Call &elem) {
    // callee
    elem.getCallee()->accept(*this);
    // arguments
    for (auto arg : elem.getArguments()) {
        arg->accept(*this);
    }
}

void Visitor::visit(CallExternal &elem) {
    // arguments for calling function
    if (elem.getArguments() != nullptr) {
        for (auto &fp : *elem.getArguments()) {
            fp.accept(*this);
        }
    }
}

void Visitor::visit(Class &elem) {
    // functions
    // TODO Think if it makes sense to represent classes at all because this output does not represent the execution
    // flow; maybe it's enough to have a list of function names in a Class?
    for (Function f : elem.getMethods()) {
        f.accept(*this);
    }
}

void Visitor::visit(Function &elem) {
    // visit FunctionParameter
    for (auto fp : elem.getParams()) {
        fp.accept(*this);
    }
    // visit Body statements
    for (auto &stmt : elem.getBody()) {
        stmt->accept(*this);
    }
}

void Visitor::visit(FunctionParameter &elem) {

}

void Visitor::visit(Group &elem) {
    elem.getExpr()->accept(*this);
}

void Visitor::visit(If &elem) {
    // condition
    elem.getCondition()->accept(*this);
    // thenBranch
    elem.getThenBranch()->accept(*this);
    // elseBranch
    elem.getElseBranch()->accept(*this);
}

void Visitor::visit(LiteralBool &elem) {

}

void Visitor::visit(LiteralInt &elem) {

}

void Visitor::visit(LiteralString &elem) {

}

void Visitor::visit(LogicalExpr &elem) {
    // left
    elem.getLeft()->accept(*this);
    // operator
    elem.getOp()->accept(*this);
    // right
    elem.getRight()->accept(*this);
}

void Visitor::visit(Operator &elem) {

}

void Visitor::visit(Return &elem) {
    elem.getValue()->accept(*this);
}

void Visitor::visit(UnaryExpr &elem) {
    // operator
    elem.getOp()->accept(*this);
    // rhs operand
    elem.getRight()->accept(*this);
}

void Visitor::visit(VarAssignm &elem) {
    elem.getValue()->accept(*this);

}

void Visitor::visit(VarDecl &elem) {
    // visit initializer
    if (elem.getInitializer() != nullptr) {
        elem.getInitializer()->accept(*this);
    }
}

void Visitor::visit(Variable &elem) {

}

void Visitor::visit(While &elem) {
    // condition
    elem.getCondition()->accept(*this);
    // then-block
    elem.getBody()->accept(*this);
}

