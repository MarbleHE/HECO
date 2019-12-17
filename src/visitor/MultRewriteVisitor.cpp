#include <iostream>
#include "../../include/visitor/MultRewriteVisitor.h"
#include "BinaryExpr.h"
#include "Block.h"
#include "../../main.h"


void MultRewriteVisitor::visit(Ast &elem) {
    Visitor::visit(elem);
}

void MultRewriteVisitor::visit(BinaryExpr &elem) {
    // if current BinaryExpr is a multiplication
    if (elem.getOp().getOperatorString() == OpSymb::getTextRepr(OpSymb::BinaryOp::multiplication)) {
        // if there is a last statement
        if (auto lastStatement = currentScope->getNthLastStatement(2)) {
            // if previous statement in scope was a BinaryExpr multiplication...
            // (Note: we need to check the penultimate statement because the statement this BinaryExpr (elem) belongs to
            // was already pushed to the scope's statement list)
            if (BinaryExpr *lastStat = lastStatement->contains(
                    new BinaryExpr(new Operator(OpSymb::BinaryOp::multiplication)))) {
                // ... then swap previousBexpLeftOp with currentBexpRightOp
                BinaryExpr::swapOperandsLeftAWithRightB(lastStat, &elem);
            }
        }
    }
    Visitor::visit(elem);
}

void MultRewriteVisitor::visit(Block &elem) {
    Visitor::visit(elem);
}

void MultRewriteVisitor::visit(Call &elem) {
    Visitor::visit(elem);
}

void MultRewriteVisitor::visit(CallExternal &elem) {
    Visitor::visit(elem);
}

void MultRewriteVisitor::visit(Class &elem) {
    Visitor::visit(elem);
}

void MultRewriteVisitor::visit(Function &elem) {
    Visitor::visit(elem);
}

void MultRewriteVisitor::visit(FunctionParameter &elem) {
    Visitor::visit(elem);
}

void MultRewriteVisitor::visit(Group &elem) {
    Visitor::visit(elem);
}

void MultRewriteVisitor::visit(If &elem) {
    Visitor::visit(elem);
}

void MultRewriteVisitor::visit(LiteralBool &elem) {
    Visitor::visit(elem);
}

void MultRewriteVisitor::visit(LiteralInt &elem) {
    Visitor::visit(elem);
}

void MultRewriteVisitor::visit(LiteralString &elem) {
    Visitor::visit(elem);
}

void MultRewriteVisitor::visit(LogicalExpr &elem) {
    Visitor::visit(elem);
}

void MultRewriteVisitor::visit(Operator &elem) {
    Visitor::visit(elem);
}

void MultRewriteVisitor::visit(Return &elem) {
    Visitor::visit(elem);
}

void MultRewriteVisitor::visit(UnaryExpr &elem) {
    Visitor::visit(elem);
}

void MultRewriteVisitor::visit(VarAssignm &elem) {
    Visitor::visit(elem);
}

void MultRewriteVisitor::visit(VarDecl &elem) {
    Visitor::visit(elem);
}

void MultRewriteVisitor::visit(Variable &elem) {
    Visitor::visit(elem);
}

void MultRewriteVisitor::visit(While &elem) {
    Visitor::visit(elem);
}


