
#include "../../include/ast/FunctionParameter.h"

#include <utility>

FunctionParameter::FunctionParameter(const std::string &identifier,
                                     std::string datatype) : Variable(identifier), datatype(std::move(datatype)) {}

void to_json(json &j, const FunctionParameter &param) {
    j = {
            {"type",       "FunctionParameter"},
            {"identifier", param.identifier},
            {"datatype",   param.datatype}
    };
}

json FunctionParameter::toJson() const {
    json j;
    //to_json(j, *this);
    return j;
}

void FunctionParameter::accept(Visitor &v) {
    v.visit(*this);
}
