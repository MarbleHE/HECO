#include "../../include/ast/Function.h"

#include <utility>
#include <iostream>
#include <LiteralInt.h>
#include <BinaryExpr.h>


void Function::addParameter(FunctionParameter *param) {
    this->params.emplace_back(*param);
}

Function::Function(std::string name, std::vector<AbstractStatement *> pt) : name(std::move(name)),
                                                                            body(std::move(pt)) {

}

Function::Function(std::string name) : name(std::move(name)) {}

Function::Function(const Function &func) : Node(func) {
    // copy 'identifier'
    this->name = std::string(func.name);
    // copy 'params'
    for (const auto &p : func.params) this->params.push_back(p);
    // copy 'body'
    this->body.reserve(func.body.size());
    for (const auto &e : func.body) this->body.emplace_back(e);
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

void Function::accept(Visitor &v) {
    v.visit(*this);
}

const std::vector<AbstractStatement *> &Function::getBody() const {
    return body;
}

void to_json(json &j, const Function &func) {
    j = {
            {"type",   func.getNodeName()},
            {"params", func.getParams()},
            {"body",   func.getBody()}};
}

json Function::toJson() const {
    json j = {
            {"type",   getNodeName()},
            {"params", params},
            {"body",   getBody()}
    };
    return j;
}

std::string Function::getNodeName() const {
    return "Function";
}

void Function::setParams(std::vector<FunctionParameter> *paramsVec) {
    this->params = *paramsVec;
}

