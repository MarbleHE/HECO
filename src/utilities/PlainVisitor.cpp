#include "ast_opt/utilities/PlainVisitor.h"

#include "ast_opt/ast/AbstractExpression.h"
#include "ast_opt/ast/AbstractNode.h"
#include "ast_opt/ast/AbstractStatement.h"
#include "ast_opt/ast/AbstractTarget.h"
#include "ast_opt/ast/Assignment.h"
#include "ast_opt/ast/BinaryExpression.h"
#include "ast_opt/ast/Block.h"
#include "ast_opt/ast/Call.h"
#include "ast_opt/ast/ExpressionList.h"
#include "ast_opt/ast/For.h"
#include "ast_opt/ast/Function.h"
#include "ast_opt/ast/FunctionParameter.h"
#include "ast_opt/ast/If.h"
#include "ast_opt/ast/IndexAccess.h"
#include "ast_opt/ast/Literal.h"
#include "ast_opt/ast/OperatorExpression.h"
#include "ast_opt/ast/Return.h"
#include "ast_opt/ast/TernaryOperator.h"
#include "ast_opt/ast/UnaryExpression.h"
#include "ast_opt/ast/Variable.h"
#include "ast_opt/ast/VariableDeclaration.h"

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
