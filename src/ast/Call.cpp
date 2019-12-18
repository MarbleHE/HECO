#include "../../include/ast/Call.h"

#include <utility>

json Call::toJson() const {
    json j = {{"type",      getNodeName()},
              {"arguments", this->arguments},
              {"callee",    this->callee->toJson()}};
    return j;
}

void Call::accept(Visitor &v) {
    v.visit(*this);
}

Call::Call(AbstractExpr *callee) : callee(callee) {
}

Call::Call(AbstractExpr *callee, std::vector<FunctionParameter *> arguments) : callee(callee),
                                                                               arguments(std::move(arguments)) {}

AbstractExpr *Call::getCallee() const {
    return callee;
}

const std::vector<FunctionParameter *> &Call::getArguments() const {
    return arguments;
}

std::string Call::getNodeName() const {
    return "Call";
}

Call::~Call() {
    delete callee;
}
