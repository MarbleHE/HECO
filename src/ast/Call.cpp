#include <typeindex>
#include <sstream>
#include <utility>
#include "Call.h"
#include "Function.h"
#include "AbstractLiteral.h"
#include "LiteralInt.h"
#include "LiteralBool.h"
#include "LiteralString.h"

json Call::toJson() const {
  json j = {{"type", getNodeName()},
            {"arguments", this->arguments},
            {"function", this->func->toJson()}};
  return j;
}

void Call::accept(Visitor &v) {
  v.visit(*this);
}

const std::vector<FunctionParameter *> &Call::getArguments() const {
  return arguments;
}

std::string Call::getNodeName() const {
  return "Call";
}

Call::~Call() {
  delete func;
}

Call::Call(Function *func) : func(func) {
}

Call::Call(std::vector<FunctionParameter *> arguments, Function *func) : func(func), arguments(std::move(arguments)) {
}

Function *Call::getFunc() const {
  return func;
}

AbstractNode *Call::clone(bool keepOriginalUniqueNodeId) {
  std::vector<FunctionParameter *> clonedArgs;
  for (auto &arg : getArguments()) {
    clonedArgs.push_back(arg->clone(keepOriginalUniqueNodeId)->castTo<FunctionParameter>());
  }
  auto clonedNode = static_cast<AbstractStatement *>(
      new Call(clonedArgs, this->getFunc()->clone(keepOriginalUniqueNodeId)->castTo<Function>()));

  if (keepOriginalUniqueNodeId) clonedNode->setUniqueNodeId(this->AbstractExpr::getUniqueNodeId());
  if (this->AbstractExpr::isReversed) clonedNode->swapChildrenParents();
  return clonedNode;
}
