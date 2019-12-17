#include "../../include/ast/LiteralString.h"

LiteralString::LiteralString(const std::string &value) : value(value) {}

json LiteralString::toJson() const {
    json j;
    j["type"] = getNodeName();
    j["value"] = this->value;
    return j;
}

void LiteralString::accept(Visitor &v) {
    v.visit(*this);
}

const std::string &LiteralString::getValue() const {
    return value;
}

std::string LiteralString::getNodeName() const {
    return "LiteralString";
}
