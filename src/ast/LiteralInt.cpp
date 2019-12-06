
#include <iostream>
#include "../../include/ast/LiteralInt.h"

LiteralInt::LiteralInt(int value) : value(value) {}

void LiteralInt::print() {
    printf("LiteralInt { \n\tvalue: %d \n}", value);
}
