#include "../../include/ast/LiteralBool.h"

LiteralBool::LiteralBool(bool value) : value(value) {}


json LiteralBool::toJson() const {
    json j;
    j["type"] = "LiteralBool";
    j["value"] = this->value;
    return j;
}
