#include "Function.h"
#include <utility>
#include <iostream>
#include "LiteralInt.h"
#include "BinaryExpr.h"
#include "Block.h"
#include "Return.h"
#include "AbstractStatement.h"

void Function::addParameter(FunctionParameter *param) {
  if (children.empty()) {
    addChild(new ParameterList({param}));
  } else {
    dynamic_cast<ParameterList *>(children[0])->addChild(param);
  }
}

Function::Function(std::string name, Block *pt) : name(std::move(name)) {
  addChild(new ParameterList());
  addChild(pt);
}

Function::Function(std::string name) : name(std::move(name)) {
  addChild(new ParameterList());
  addChild(new Block());
}

Function::Function(std::string functionName, ParameterList *functionParameters,
                   Block *functionStatements) : name(std::move(functionName)) {
  addChild(functionParameters);
  addChild(functionStatements);

}

void Function::addStatement(AbstractStatement *statement) {
  if (children.size() < 1) addChild(new ParameterList());
  getBody()->addChild(statement);
}

const std::string &Function::getName() const {
  return name;
}

void Function::accept(Visitor &v) {
  v.visit(*this);
}

std::vector<AbstractStatement *> Function::getBodyStatements() const {
  return *getBody()->getStatements();
}

void to_json(json &j, const Function &func) {
  j = {
      {"type", func.getNodeName()},
      {"name", func.getName()},
      {"params", func.getParameters()},
      {"body", func.getBodyStatements()}};
}

json Function::toJson() const {
  json j = {
      {"type", getNodeName()},
      {"name", getName()},
      {"params", getParameters()},
      {"body", getBodyStatements()}
  };
  return j;
}

std::string Function::getNodeName() const {
  return "Function";
}

void Function::setParameterList(ParameterList *paramsVec) {
  if (!children.empty()) this->removeChild(children[0]);
  children[0] = paramsVec;

}

Function *Function::clone(bool keepOriginalUniqueNodeId) {
  auto clonedParams = getParameterList() ? getParameterList()->clone(keepOriginalUniqueNodeId) : nullptr;
  auto clonedBody = getBody() ? getBody()->clone(keepOriginalUniqueNodeId) : nullptr;
  auto clonedNode = new Function(this->getName(), clonedParams, clonedBody);
  if (keepOriginalUniqueNodeId) clonedNode->setUniqueNodeId(this->getUniqueNodeId());
  if (this->isReversed) clonedNode->swapChildrenParents();
  return clonedNode;
}

std::vector<FunctionParameter *> Function::getParameters() const {
  return children.empty() ? std::vector<FunctionParameter *>() : dynamic_cast<ParameterList *>(children[0])
      ->getParameters();
}
ParameterList *Function::getParameterList() const {
  return children.empty() ? nullptr : dynamic_cast<ParameterList *>(children[0]);
}
Block *Function::getBody() const {
  return dynamic_cast<Block *>(getChildAtIndex(1));
}
Function::Function(std::string functionName,
                   std::vector<FunctionParameter *> functionParameters,
                   std::vector<AbstractStatement *> functionStatements) : name(std::move(functionName)) {
  addChild(new ParameterList(functionParameters));
  addChild(new Block(&functionStatements));
}
int Function::getMaxNumberChildren() {
  return 2;
}
bool Function::supportsCircuitMode() {
  return true;
}
