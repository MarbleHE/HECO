#include <utility>
#include <vector>
#include "FunctionParameter.h"
#include "CallExternal.h"

json CallExternal::toJson() const {
  json j = {
      {"type", getNodeType()},
      {"functionName", this->functionName},
  };
  // only include the field 'arguments' in JSON output if it has any value
  if (!this->arguments.empty()) {
    j["arguments"] = this->arguments;
  }
  return j;
}

CallExternal::CallExternal(std::string functionName, std::vector<FunctionParameter *> arguments)
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

std::string CallExternal::getNodeType() const {
  return "CallExternal";
}

AbstractNode *CallExternal::clone(bool keepOriginalUniqueNodeId) {
  std::vector<FunctionParameter *> args;
  for (auto &fp : this->getArguments()) {
    args.push_back(fp->clone(keepOriginalUniqueNodeId)->castTo<FunctionParameter>());
  }
  auto clonedCallExt = new CallExternal(getFunctionName(), args);
  return clonedCallExt;
}

const std::vector<FunctionParameter *> &CallExternal::getArguments() const {
  return arguments;
}
