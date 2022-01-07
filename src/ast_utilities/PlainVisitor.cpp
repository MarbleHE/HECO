#include "abc/ast_utilities/PlainVisitor.h"

#include "abc/ast/AbstractExpression.h"
#include "abc/ast/AbstractNode.h"
#include "abc/ast/AbstractStatement.h"
#include "abc/ast/AbstractTarget.h"
#include "abc/ast/Assignment.h"
#include "abc/ast/BinaryExpression.h"
#include "abc/ast/Block.h"
#include "abc/ast/Call.h"
#include "abc/ast/ExpressionList.h"
#include "abc/ast/For.h"
#include "abc/ast/Function.h"
#include "abc/ast/FunctionParameter.h"
#include "abc/ast/If.h"
#include "abc/ast/IndexAccess.h"
#include "abc/ast/Literal.h"
#include "abc/ast/OperatorExpression.h"
#include "abc/ast/Return.h"
#include "abc/ast/TernaryOperator.h"
#include "abc/ast/UnaryExpression.h"
#include "abc/ast/Variable.h"
#include "abc/ast/VariableDeclaration.h"

void PlainVisitor::visit(BinaryExpression &elem) {
  visitChildren(elem);
}

void PlainVisitor::visit(Block &elem) {
  visitChildren(elem);
}

void PlainVisitor::visit(Call &elem) {
  visitChildren(elem);
}

void PlainVisitor::visit(ExpressionList &elem) {
  visitChildren(elem);
}

void PlainVisitor::visit(For &elem) {
  visitChildren(elem);
}

void PlainVisitor::visit(Function &elem) {
  visitChildren(elem);
}

void PlainVisitor::visit(FunctionParameter &elem) {
  visitChildren(elem);
}

void PlainVisitor::visit(If &elem) {
  visitChildren(elem);
}

void PlainVisitor::visit(IndexAccess &elem) {
  visitChildren(elem);
}

void PlainVisitor::visit(LiteralBool &elem) {
  visitChildren(elem);
}

void PlainVisitor::visit(LiteralChar &elem) {
  visitChildren(elem);
}

void PlainVisitor::visit(LiteralInt &elem) {
  visitChildren(elem);
}

void PlainVisitor::visit(LiteralFloat &elem) {
  visitChildren(elem);
}

void PlainVisitor::visit(LiteralDouble &elem) {
  visitChildren(elem);
}

void PlainVisitor::visit(LiteralString &elem) {
  visitChildren(elem);
}

void PlainVisitor::visit(OperatorExpression &elem) {
  visitChildren(elem);
}

void PlainVisitor::visit(Return &elem) {
  visitChildren(elem);
}

void PlainVisitor::visit(TernaryOperator &elem) {
  visitChildren(elem);
}

void PlainVisitor::visit(UnaryExpression &elem) {
  visitChildren(elem);
}

void PlainVisitor::visit(Assignment &elem) {
  visitChildren(elem);
}

void PlainVisitor::visit(VariableDeclaration &elem) {
  visitChildren(elem);
}

void PlainVisitor::visit(Variable &elem) {
  visitChildren(elem);
}

void PlainVisitor::visitChildren(AbstractNode &elem) {
  for (auto &c : elem) {
    c.accept(*this);
  }
}
