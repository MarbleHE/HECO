#include <iostream>
#include "../../include/ast/AbstractExpr.h"

std::string AbstractExpr::toString() const {
    return this->toJson().dump();
}

json AbstractExpr::toJson() const {
    return json({"type", "AbstractExpr"});
}

std::ostream &operator<<(std::ostream &outs, const AbstractExpr &obj) {
    return outs << obj.toString();
}
