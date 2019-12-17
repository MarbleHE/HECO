#include "../../include/ast/Group.h"

json Group::toJson() const {
    json j;
    j["type"] = getNodeName();
    j["expr"] = this->expr->toJson();
    return j;
}


void Group::accept(Visitor &v) {
    v.visit(*this);
}

std::string Group::getNodeName() const {
    return "Group";
}

Group::Group(AbstractExpr *expr) : expr(expr) {}

AbstractExpr *Group::getExpr() const {
    return expr;
}

Group::~Group() {
    delete expr;
}

