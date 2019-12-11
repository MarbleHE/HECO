
#include <iostream>
#include "../../include/ast/LiteralInt.h"


LiteralInt::LiteralInt(int value) : value(value) {}


json LiteralInt::toJson() const {
    json j;
    j["type"] = "LiteralInt";
    j["value"] = this->value;
    return j;
}

int LiteralInt::getValue() const {
    return value;
}
