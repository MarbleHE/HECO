#include "ast_opt/ast/Assignment.h"
#include "ast_opt/ast/BinaryExpression.h"
#include "ast_opt/ast/Block.h"
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

void ScopedVisitor::visit(BinaryExpression & elem) {
  for(auto& c : elem) {
    c.accept(*this);
  }
}

void ScopedVisitor::visit(Block & elem) {
  for(auto& c : elem) {
    c.accept(*this);
  }
}

void ScopedVisitor::visit(ExpressionList & elem) {
  for(auto& c : elem) {
    c.accept(*this);
  }
}

void ScopedVisitor::visit(For & elem) {
  for(auto& c : elem) {
    c.accept(*this);
  }
}

void ScopedVisitor::visit(Function & elem) {
  for(auto& c : elem) {
    c.accept(*this);
  }
}

void ScopedVisitor::visit(FunctionParameter & elem) {
  for(auto& c : elem) {
    c.accept(*this);
  }
}

void ScopedVisitor::visit(If & elem) {
  for(auto& c : elem) {
    c.accept(*this);
  }
}

void ScopedVisitor::visit(IndexAccess & elem) {
  for(auto& c : elem) {
    c.accept(*this);
  }
}

void ScopedVisitor::visit(LiteralBool & elem) {
  for(auto& c : elem) {
    c.accept(*this);
  }
}

void ScopedVisitor::visit(LiteralChar & elem) {
  for(auto& c : elem) {
    c.accept(*this);
  }
}

void ScopedVisitor::visit(LiteralInt & elem) {
  for(auto& c : elem) {
    c.accept(*this);
  }
}

void ScopedVisitor::visit(LiteralFloat & elem) {
  for(auto& c : elem) {
    c.accept(*this);
  }
}

void ScopedVisitor::visit(LiteralDouble & elem) {
  for(auto& c : elem) {
    c.accept(*this);
  }
}

void ScopedVisitor::visit(LiteralString & elem) {
  for(auto& c : elem) {
    c.accept(*this);
  }
}

void ScopedVisitor::visit(OperatorExpression & elem) {
  for(auto& c : elem) {
    c.accept(*this);
  }
}

void ScopedVisitor::visit(Return & elem) {
  for(auto& c : elem) {
    c.accept(*this);
  }
}

void ScopedVisitor::visit(UnaryExpression & elem) {
  for(auto& c : elem) {
    c.accept(*this);
  }
}

void ScopedVisitor::visit(Assignment & elem) {
  for(auto& c : elem) {
    c.accept(*this);
  }
}

void ScopedVisitor::visit(VariableDeclaration & elem) {
  for (auto &c : elem) {
    c.accept(*this);
  }
}

void ScopedVisitor::visit(Variable &elem) {
  for (auto &c : elem) {
    c.accept(*this);
  }
}

ScopedVisitor::ScopedVisitor() {
  currentScope = std::make_unique<Scope>();
}

Scope &ScopedVisitor::getCurrentScope() {
  return *currentScope;
}

const Scope &ScopedVisitor::getCurrentScope() const {
  return *currentScope;
}
