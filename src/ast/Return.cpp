#include "../../include/ast/Return.h"

Return::Return(std::unique_ptr<AbstractExpr> value) : value(std::move(value)) {}

json Return::toJson() const {
    json j;
    j["type"] = "Return";
    j["value"] = this->value->toJson();
    return j;
}