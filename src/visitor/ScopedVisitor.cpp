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
#include "ast_opt/ast/UnaryExpression.h"
#include "ast_opt/ast/Variable.h"
#include "ast_opt/ast/VariableDeclaration.h"
#include "ast_opt/visitor/ScopedVisitor.h"
#include "ast_opt/utilities/Scope.h"

void ScopedVisitor::visit(BinaryExpression &elem) {
  visitChildren(elem);
}

void ScopedVisitor::visit(Block &elem) {
  enterScope(elem);
  visitChildren(elem);
  exitScope(elem);
}

void ScopedVisitor::visit(Call &elem) {
  visitChildren(elem);
}

void ScopedVisitor::visit(ExpressionList &elem) {
  visitChildren(elem);
}

void ScopedVisitor::visit(For &elem) {
  enterScope(elem);
  visitChildren(elem);
  exitScope(elem);
}

void ScopedVisitor::visit(Function &elem) {
  visitChildren(elem);
}

void ScopedVisitor::visit(FunctionParameter &elem) {
  getCurrentScope().addIdentifier(elem.getIdentifier());
  visitChildren(elem);
}

void ScopedVisitor::visit(If &elem) {
  enterScope(elem);
  visitChildren(elem);
  exitScope(elem);
}

void ScopedVisitor::visit(IndexAccess &elem) {
  visitChildren(elem);
}

void ScopedVisitor::visit(LiteralBool &elem) {
  visitChildren(elem);
}

void ScopedVisitor::visit(LiteralChar &elem) {
  visitChildren(elem);
}

void ScopedVisitor::visit(LiteralInt &elem) {
  visitChildren(elem);
}

void ScopedVisitor::visit(LiteralFloat &elem) {
  visitChildren(elem);
}

void ScopedVisitor::visit(LiteralDouble &elem) {
  visitChildren(elem);
}

void ScopedVisitor::visit(LiteralString &elem) {
  visitChildren(elem);
}

void ScopedVisitor::visit(OperatorExpression &elem) {
  visitChildren(elem);
}

void ScopedVisitor::visit(Return &elem) {
  visitChildren(elem);
}

void ScopedVisitor::visit(UnaryExpression &elem) {
  visitChildren(elem);
}

void ScopedVisitor::visit(Assignment &elem) {
  visitChildren(elem);
}

void ScopedVisitor::visit(VariableDeclaration &elem) {
  getCurrentScope().addIdentifier(elem.getTarget().getIdentifier());
  visitChildren(elem);
}

void ScopedVisitor::visit(Variable &elem) {
  visitChildren(elem);
}

void ScopedVisitor::visitChildren(AbstractNode &elem) {
  for (auto &c : elem) {
    c.accept(*this);
  }
}

Scope &ScopedVisitor::getCurrentScope() {
  if (currentScope==nullptr) {
    throw std::runtime_error("Cannot return non-existent scope!");
  }
  return *currentScope;
}

const Scope &ScopedVisitor::getCurrentScope() const {
  if (currentScope==nullptr) {
    throw std::runtime_error("Cannot return non-existent scope!");
  }
  return *currentScope;
}

void ScopedVisitor::enterScope(AbstractNode &node) {
  if (rootScope==nullptr) {
    // no scope created yet: create root scope and also set it as current scope
    rootScope = std::make_unique<Scope>(node);
    std::for_each(predeclaredVariables.begin(), predeclaredVariables.end(), [&](const std::string &id) {
      rootScope->addIdentifier(id);
    });
    currentScope = rootScope.get();
  } else {
    // create nested scope with current scope as parent
    currentScope = Scope::createNestedScope(getCurrentScope(), node);
  }
}

void ScopedVisitor::exitScope(AbstractNode &node) {
  if (currentScope) {
    currentScope = &currentScope->getParentScope();
  } else {
    throw std::runtime_error("Cannot leave non-existent scope. "
                             "Did you forget to call enterScope(...)?");
  }
}

void ScopedVisitor::setPredeclaredVariables(std::vector<std::string> &alreadyDeclaredVariables) {
  ScopedVisitor::predeclaredVariables = std::move(alreadyDeclaredVariables);
}
