#include "Function.h"
#include <utility>
#include <iostream>
#include "LiteralInt.h"
#include "BinaryExpr.h"
#include "Return.h"

void Function::addParameter(FunctionParameter *param) {
  this->params.emplace_back(param);
}

Function::Function(std::string name, std::vector<AbstractStatement *> pt) : name(std::move(name)), body(std::move(pt)) {
  for (auto &stmt : getBody()) {
    auto previous = *(&stmt - 1);
    auto next = *(&stmt + 1);
    if (previous != nullptr) stmt->addParent(previous);
    if (next != nullptr) stmt->addChild(next);
  }
}

Function::Function(std::string name) : name(std::move(name)) {
}

Function::Function(std::string functionName, std::vector<FunctionParameter *> functionParameters,
                   std::vector<AbstractStatement *> functionStatements) : name(std::move(functionName)),
                                                                          params(std::move(functionParameters)),
                                                                          body(std::move(functionStatements)) {}

void Function::addStatement(AbstractStatement *statement) {
  this->body.emplace_back(statement);
}

const std::string &Function::getName() const {
  return name;
}

void Function::accept(Visitor &v) {
  v.visit(*this);
}

const std::vector<AbstractStatement *> &Function::getBody() const {
  return body;
}

void to_json(json &j, const Function &func) {
  j = {
      {"type",   func.getNodeName()},
      {"name",   func.getName()},
      {"params", func.getParams()},
      {"body",   func.getBody()}};
}

json Function::toJson() const {
  json j = {
      {"type",   getNodeName()},
      {"name",   getName()},
      {"params", params},
      {"body",   getBody()}
  };
  return j;
}

std::string Function::getNodeName() const {
  return "Function";
}

void Function::setParams(std::vector<FunctionParameter *> paramsVec) {
  this->params = std::move(paramsVec);
}

std::vector<Literal *> Function::evaluate(Ast &ast) {
  for (size_t i = 0; i < this->getBody().size(); i++) {
    auto currentStatement = this->getBody().at(i);
    // last statement: check if it is a Return statement
    if (i == this->getBody().size() - 1) {
      if (auto retStmt = dynamic_cast<Return *>(currentStatement)) {
        return retStmt->evaluate(ast);
      }
    }
    (void) currentStatement->evaluate(ast);
  }
  return std::vector<Literal *>();
}

Node *Function::createClonedNode(bool keepOriginalUniqueNodeId) {
  std::vector<FunctionParameter *> clonedParams;
  for (auto &fp : this->getParams()) {
    clonedParams.push_back(fp->cloneRecursiveDeep(keepOriginalUniqueNodeId)->castTo<FunctionParameter>());
  }
  std::vector<AbstractStatement *> clonedBody;
  for (auto &statement : this->getBody()) {
    clonedBody.push_back(statement->cloneRecursiveDeep(keepOriginalUniqueNodeId)->castTo<AbstractStatement>());
  }
  return new Function(this->getName(), clonedParams, clonedBody);
}

const std::vector<FunctionParameter *> &Function::getParams() const {
  return params;
}
