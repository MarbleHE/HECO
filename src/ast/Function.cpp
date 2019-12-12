#include "../../include/ast/Function.h"

#include <utility>
#include <iostream>
#include <LiteralInt.h>
#include <BinaryExpr.h>

void Function::addParameter(const FunctionParameter &param) {
    this->params.push_back(param);
}

Function::Function(std::string name, std::vector<std::unique_ptr<AbstractStatement>> pt) : name(std::move(name)),
                                                                                           body(std::move(pt)) {
}

Function::Function(std::string name) : name(std::move(name)) {}

Function::Function(const Function &func) {
    // copy 'name'
    this->name = std::string(func.name);
    // copy 'params'
    for (const auto &p : func.params) this->params.push_back(p);
    // copy 'body'
    this->body.reserve(func.body.size());
    for (const auto &e : func.body) this->body.push_back(std::make_unique<AbstractStatement>(*e));
}

void to_json(json &j, const Function &func) {
    j = {
            {"type",   "Function"},
            {"params", func.getParams()},
            {"body",   func.getBody()}

//            {"params", func.getParams()},
//            {"body",   func.getBody()}
    };
}

json Function::toJson() const {
    json j = {
            {"type",   "Function"},
            {"params", params},
            {"body",   body}
    };
    return j;
}

Function::Function() {

}

void Function::addStatement(AbstractStatement *statement) {
    this->body.emplace_back(statement);
}

const std::string &Function::getName() const {
    return name;
}

const std::vector<FunctionParameter> &Function::getParams() const {
    return params;
}

const std::vector<std::unique_ptr<AbstractStatement>> &Function::getBody() const {
    return body;
}


