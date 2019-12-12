#include <iostream>
#include "../../include/ast/AbstractExpr.h"

std::string AbstractExpr::toString() const {
    return this->toJson().dump();
}

json AbstractExpr::toJson() const {
    return json({"type", "AbstractExpr"});
}

void AbstractExpr::accept(Visitor &v) {
    std::cout << "This shouldn't be executed!" << std::endl;
}

std::ostream &operator<<(std::ostream &outs, const AbstractExpr &obj) {
    return outs << obj.toString();
}
