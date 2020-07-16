#include <ast_opt/visitor/Visitor.h>

void ScopedVisitor::visit(BinaryExpression &elem) {}

void ScopedVisitor::visit(Block &elem) {}

void ScopedVisitor::visit(For &elem) {}

void ScopedVisitor::visit(If &elem) {}

void ScopedVisitor::visit(LiteralBool &elem) {}

void ScopedVisitor::visit(LiteralChar &elem) {}

void ScopedVisitor::visit(LiteralInt &elem) {}

void ScopedVisitor::visit(LiteralFloat &elem) {}

void ScopedVisitor::visit(LiteralDouble &elem) {}

void ScopedVisitor::visit(LiteralString &elem) {}

void ScopedVisitor::visit(UnaryExpression &elem) {}

void ScopedVisitor::visit(VariableAssignment &elem) {}

void ScopedVisitor::visit(VariableDeclaration &elem) {}

void ScopedVisitor::visit(Variable &elem) {}