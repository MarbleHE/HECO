#include "../../include/ast/Function.h"

#include <utility>
#include <iostream>
#include <LiteralInt.h>
#include <BinaryExpr.h>
#include "Return.h"

void Function::addParameter(FunctionParameter* param) {
  this->params.emplace_back(*param);
}

Function::Function(std::string name, std::vector<AbstractStatement*> pt) : name(std::move(name)), body(std::move(pt)) {
  for (auto &stmt : getBody()) {
    auto previous = *(&stmt - 1);
    auto next = *(&stmt + 1);
    if (previous != nullptr) {
      stmt->addParent(previous);
    }
    if (next != nullptr) {
      stmt->addChild(next);
    }
  }
}

Function::Function(std::string name) : name(std::move(name)) {
}

void Function::addStatement(AbstractStatement* statement) {
  if (this->getBody().empty()) {
    this->addChild(statement);
    statement->addParent(this);
  } else {
    auto lastStatement = this->getBody().back();
    lastStatement->addChild(statement);
    statement->addParent(lastStatement);
  }
  this->body.emplace_back(statement);
}

const std::string &Function::getName() const {
  return name;
}

const std::vector<FunctionParameter> &Function::getParams() const {
  return params;
}

void Function::accept(Visitor &v) {
  v.visit(*this);
}

const std::vector<AbstractStatement*> &Function::getBody() const {
  return body;
}

void to_json(json &j, const Function &func) {
  j = {
      {"type", func.getNodeName()},
      {"name", func.getName()},
      {"params", func.getParams()},
      {"body", func.getBody()}};
}

json Function::toJson() const {
  json j = {
      {"type", getNodeName()},
      {"name", getName()},
      {"params", params},
      {"body", getBody()}
  };
  return j;
}

std::string Function::getNodeName() const {
  return "Function";
}

void Function::setParams(std::vector<FunctionParameter>* paramsVec) {
  this->params = *paramsVec;
}

Function::Function(std::string name, std::vector<FunctionParameter> params,
                   std::vector<AbstractStatement*> body) : name(std::move(name)),
                                                           params(std::move(params)),
                                                           body(std::move(body)) {}

Literal* Function::evaluate(Ast &ast) {
  for (int i = 0; i < this->getBody().size(); i++) {
    auto currentStatement = this->getBody().at(i);
    // last statement: check if it is a Return statement
    if (i == this->getBody().size() - 1) {
      if (auto retStmt = dynamic_cast<Return*>(currentStatement)) {
        return retStmt->evaluate(ast);
      }
    }
    (void) currentStatement->evaluate(ast);
  }
  return nullptr;
}

