#include <utility>
#include <vector>
#include "FunctionParameter.h"
#include "CallExternal.h"

json CallExternal::toJson() const {
  json j = {
      {"type", getNodeName()},
      {"functionName", this->functionName},
  };
  // only include the field 'arguments' in JSON output if it has any value
  if (!this->arguments.empty()) {
    j["arguments"] = this->arguments;
  }
  return j;
}

CallExternal::CallExternal(std::string functionName, std::vector<FunctionParameter*> arguments)
    : functionName(std::move(functionName)) {
  this->arguments = std::move(arguments);
}

CallExternal::CallExternal(std::string functionName) : functionName(std::move(std::move(functionName))) {}

void CallExternal::accept(Visitor &v) {
  v.visit(*this);
}

const std::string &CallExternal::getFunctionName() const {
  return functionName;
}

std::string CallExternal::getNodeName() const {
  return "CallExternal";
}

Literal* CallExternal::evaluate(Ast &ast) {
  throw std::runtime_error(
      "evaluateAst(Ast &ast) not implemented for class CallExternal yet! Consider using Call instead.");
}

Node* CallExternal::createClonedNode(bool keepOriginalUniqueNodeId) {
  std::vector<FunctionParameter*> args;
  for (auto &fp : this->getArguments()) {
    args.push_back(fp->cloneRecursiveDeep(keepOriginalUniqueNodeId)->castTo<FunctionParameter>());
  }
  // this is required because CallExternal inherits from both AbstractStatement and AbstractExpr
  // -> ambiguous base class
  return static_cast<AbstractStatement*>(new CallExternal("abc", args));
}
const std::vector<FunctionParameter*> &CallExternal::getArguments() const {
  return arguments;
}
