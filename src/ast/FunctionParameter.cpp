#include "../../include/ast/FunctionParameter.h"

#include <utility>

FunctionParameter::FunctionParameter(const std::string &identifier,
                                     std::string datatype) : Variable(identifier), datatype(std::move(datatype)) {}

void to_json(json &j, const FunctionParameter &funcParam) {
    j = {
            {"type",       funcParam.getNodeName()},
            {"identifier", funcParam.getIdentifier()},
            {"datatype",   funcParam.getDatatype()}
    };
}

json FunctionParameter::toJson() const {
    json j;
    //to_json(j, *this); // TODO implement me!
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
