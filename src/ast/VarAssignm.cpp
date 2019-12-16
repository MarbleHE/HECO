
#include "../../include/ast/VarAssignm.h"

json VarAssignm::toJson() const {
    json j;
    j["type"] = getNodeName();
    j["identifier"] = this->identifier;
    j["value"] = this->value->toJson();
    return j;
}

VarAssignm::VarAssignm(const std::string &identifier, AbstractExpr *value) : identifier(identifier), value(value) {}

void VarAssignm::accept(Visitor &v) {
    v.visit(*this);
}

const std::string &VarAssignm::getIdentifier() const {
    return identifier;
}

AbstractExpr *VarAssignm::getValue() const {
    return value;
}

std::string VarAssignm::getNodeName() const {
    return "VarAssignm";
}
