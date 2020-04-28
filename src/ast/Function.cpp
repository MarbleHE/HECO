#include <utility>
#include <iostream>
#include "ast_opt/ast/Function.h"
#include "ast_opt/ast/LiteralInt.h"
#include "ast_opt/ast/ArithmeticExpr.h"
#include "ast_opt/ast/Block.h"
#include "ast_opt/ast/Return.h"
#include "ast_opt/ast/AbstractStatement.h"

void Function::addParameter(FunctionParameter *param) {
  if (children.empty()) {
    addChild(new ParameterList({param}));
  } else {
    dynamic_cast<ParameterList *>(getChildAtIndex(0))->addChild(param);
  }
}

Function::Function(std::string name, Block *pt) : name(std::move(name)) {
  addChildren({new ParameterList(), pt});
}

Function::Function(std::string name) : name(std::move(name)) {
  addChildren({new ParameterList(), new Block()});
}

Function::Function(std::string functionName, ParameterList *functionParameters,
                   Block *functionStatements) : name(std::move(functionName)) {
  addChildren({functionParameters, functionStatements});

}

void Function::addStatement(AbstractStatement *statement) {
  if (children.empty()) addChild(new ParameterList());
  getBody()->addChild(statement);
}

const std::string &Function::getName() const {
  return name;
}

void Function::accept(Visitor &v) {
  v.visit(*this);
}

std::vector<AbstractStatement *> Function::getBodyStatements() const {
  return getBody()->getStatements();
}

void to_json(json &j, const Function &func) {
  j = {
      {"type", func.getNodeType()},
      {"name", func.getName()},
      {"params", func.getParameters()},
      {"body", func.getBodyStatements()}};
}

json Function::toJson() const {
  json j = {
      {"type", getNodeType()},
      {"name", getName()},
      {"params", getParameters()},
      {"body", getBodyStatements()}
  };
  return j;
}

std::string Function::getNodeType() const {
  return "Function";
}

void Function::setParameterList(ParameterList *paramsVec) {
  if (!children.empty()) this->removeChild(getChildAtIndex(0), false);
  addChild(paramsVec);
}

Function *Function::clone(bool keepOriginalUniqueNodeId) {
  auto clonedParams = getParameterList() ? getParameterList()->clone(keepOriginalUniqueNodeId) : nullptr;
  auto clonedBody = getBody() ? getBody()->clone(keepOriginalUniqueNodeId) : nullptr;
  auto clonedNode = new Function(this->getName(), clonedParams, clonedBody);
  clonedNode->updateClone(keepOriginalUniqueNodeId, this);
  return clonedNode;
}

std::vector<FunctionParameter *> Function::getParameters() const {
  return children.empty() ? std::vector<FunctionParameter *>()
                          : reinterpret_cast<ParameterList *>(getChildAtIndex(0))->getParameters();
}

ParameterList *Function::getParameterList() const {
  return reinterpret_cast<ParameterList *>(getChildAtIndex(0));
}

Block *Function::getBody() const {
  return reinterpret_cast<Block *>(getChildAtIndex(1));
}

Function::Function(std::string functionName,
                   std::vector<FunctionParameter *> functionParameters,
                   std::vector<AbstractStatement *> functionStatements) : name(std::move(functionName)) {
  addChildren({new ParameterList(std::move(functionParameters)),
               new Block(std::move(functionStatements))});
}

int Function::getMaxNumberChildren() {
  return 2;
}

bool Function::supportsCircuitMode() {
  return true;
}

std::string Function::toString(bool printChildren) const {
  return AbstractNode::generateOutputString(printChildren, {getName()});
}
