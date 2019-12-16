
#include <iostream>
#include "../../include/visitor/MultRewriteVisitor.h"

void MultRewriteVisitor::visit(Ast &elem) {
    // do sth 'MultRewriteVisitor' related
    std::cout << __func__ << " is visiting Ast element." << std::endl;
    // continue traversal
    //Visitor::visit(elem); // TODO requires that visit order logic is in Visitor class
}

void MultRewriteVisitor::visit(BinaryExpr &elem) {
    // NOTE only rewrite mult if it is in the same scope
    // Determine by level?

}

void MultRewriteVisitor::visit(Block &elem) {

}

void MultRewriteVisitor::visit(Call &elem) {

}

void MultRewriteVisitor::visit(CallExternal &elem) {

}

void MultRewriteVisitor::visit(Class &elem) {

}

void MultRewriteVisitor::visit(Function &elem) {

}

void MultRewriteVisitor::visit(FunctionParameter &elem) {

}

void MultRewriteVisitor::visit(Group &elem) {

}

void MultRewriteVisitor::visit(If &elem) {

}

void MultRewriteVisitor::visit(LiteralBool &elem) {

}

void MultRewriteVisitor::visit(LiteralInt &elem) {

}

void MultRewriteVisitor::visit(LiteralString &elem) {

}

void MultRewriteVisitor::visit(LogicalExpr &elem) {

}

void MultRewriteVisitor::visit(Operator &elem) {

}

void MultRewriteVisitor::visit(Return &elem) {

}

void MultRewriteVisitor::visit(UnaryExpr &elem) {

}

void MultRewriteVisitor::visit(VarAssignm &elem) {

}

void MultRewriteVisitor::visit(VarDecl &elem) {

}

void MultRewriteVisitor::visit(Variable &elem) {

}

void MultRewriteVisitor::visit(While &elem) {

}
