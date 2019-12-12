#include "../../include/ast/Group.h"

json Group::toJson() const {
    json j;
    j["type"] = "Group";
    j["expr"] = this->expr->toJson();
    return j;
}

Group::Group(std::unique_ptr<AbstractExpr> expr) : expr(std::move(expr)) {}

void Group::accept(Visitor &v) {
    v.visit(*this);
}
