#include "ast_opt/visitor/Runtime/RuntimeVisitor.h"

void SpecialRuntimeVisitor::visit(BinaryExpression &elem) {
  ScopedVisitor::visit(elem);
}

void SpecialRuntimeVisitor::visit(Block &elem) {
  ScopedVisitor::visit(elem);
}

void SpecialRuntimeVisitor::visit(Call &elem) {
  ScopedVisitor::visit(elem);
}

void SpecialRuntimeVisitor::visit(ExpressionList &elem) {
  ScopedVisitor::visit(elem);
}

void SpecialRuntimeVisitor::visit(For &elem) {
  ScopedVisitor::visit(elem);
}

void SpecialRuntimeVisitor::visit(Function &elem) {
  ScopedVisitor::visit(elem);
}

void SpecialRuntimeVisitor::visit(FunctionParameter &elem) {
  ScopedVisitor::visit(elem);
}

void SpecialRuntimeVisitor::visit(If &elem) {
  ScopedVisitor::visit(elem);
}

void SpecialRuntimeVisitor::visit(IndexAccess &elem) {
  ScopedVisitor::visit(elem);
}

void SpecialRuntimeVisitor::visit(LiteralBool &elem) {
  ScopedVisitor::visit(elem);
}

void SpecialRuntimeVisitor::visit(LiteralChar &elem) {
  ScopedVisitor::visit(elem);
}

void SpecialRuntimeVisitor::visit(LiteralInt &elem) {
  ScopedVisitor::visit(elem);
}

void SpecialRuntimeVisitor::visit(LiteralFloat &elem) {
  ScopedVisitor::visit(elem);
}

void SpecialRuntimeVisitor::visit(LiteralDouble &elem) {
  ScopedVisitor::visit(elem);
}

void SpecialRuntimeVisitor::visit(LiteralString &elem) {
  ScopedVisitor::visit(elem);
}

void SpecialRuntimeVisitor::visit(OperatorExpression &elem) {
  ScopedVisitor::visit(elem);
}

void SpecialRuntimeVisitor::visit(Return &elem) {
  ScopedVisitor::visit(elem);
}

void SpecialRuntimeVisitor::visit(UnaryExpression &elem) {
  ScopedVisitor::visit(elem);
}

void SpecialRuntimeVisitor::visit(Assignment &elem) {
  ScopedVisitor::visit(elem);
}

void SpecialRuntimeVisitor::visit(VariableDeclaration &elem) {
  ScopedVisitor::visit(elem);
}

void SpecialRuntimeVisitor::visit(Variable &elem) {
  ScopedVisitor::visit(elem);
}
