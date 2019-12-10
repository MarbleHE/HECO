
#include <iostream>
#include "../../include/ast/LiteralInt.h"

LiteralInt::LiteralInt(int value) : value(value) {}

std::string LiteralInt::toString() const {
    return std::to_string(this->value);
}
