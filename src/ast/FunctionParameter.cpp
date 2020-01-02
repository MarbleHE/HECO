
#include "../../include/ast/FunctionParameter.h"

#include <utility>

FunctionParameter::FunctionParameter(std::string datatype, AbstractExpr *value)
    : datatype(std::move(datatype)), value(value) {}

void to_json(json &j, const FunctionParameter &funcParam) {
    j = {
        {"type", funcParam.getNodeName()},
        {"value", funcParam.getValue()->toJson()},
        {"datatype", funcParam.getDatatype()}
    };
}

void to_json(json &j, const FunctionParameter *funcParam) {
    j = {
        {"type", funcParam->getNodeName()},
        {"value", funcParam->getValue()->toJson()},
        {"datatype", funcParam->getDatatype()}
    };
}

json FunctionParameter::toJson() const {
    json j = {
        {"type", getNodeName()},
        {"value", getValue()->toJson()},
        {"datatype", getDatatype()}
    };
    return j;
}

void FunctionParameter::accept(Visitor &v) {
    v.visit(*this);
}

std::string FunctionParameter::getNodeName() const {
    return "FunctionParameter";
}

const std::string &FunctionParameter::getDatatype() const {
    return datatype;
}

AbstractExpr *FunctionParameter::getValue() const {
    return value;
}
