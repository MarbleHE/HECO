#include "../../include/ast/Function.h"

void Function::addParameter(FunctionParameter param) {
    this->params.push_back(param);
}

Function::Function(std::string name, std::vector<std::unique_ptr<AbstractStatement>> pt) : name(name), body(std::move(pt)){
}
