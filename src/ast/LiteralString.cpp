
#include "../../include/ast/LiteralString.h"

LiteralString::LiteralString(const std::string &value) : value(value) {}

json LiteralString::toJson() const {
    json j;
    j["type"] = "LiteralString";
    j["value"] = this->value;
    return j;
}
