#include "../../include/ast/Call.h"

#include <utility>
#include "Function.h"

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

Call::Call(Function *func) : func(func) {}

Call::Call(std::vector<FunctionParameter *> arguments, Function *func) : func(func), arguments(std::move(arguments)) {}

Function *Call::getFunc() const {
  return func;
}
