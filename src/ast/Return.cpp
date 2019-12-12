#include "../../include/ast/Return.h"


json Return::toJson() const {
    json j;
    j["type"] = "Return";
    j["value"] = this->value->toJson();
    return j;
}

Return::Return(AbstractExpr *value) : value(value) {}

void Return::accept(Visitor &v) {
    v.visit(*this);
}
