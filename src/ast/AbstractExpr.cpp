#include <iostream>
#include "../../include/ast/AbstractExpr.h"

std::string AbstractExpr::toString() const {
    return std::string("AbstractExpr");
}

std::ostream &operator<<(std::ostream &outs, const AbstractExpr &obj) {
    return outs << obj.toString();
}
