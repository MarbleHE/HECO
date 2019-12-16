#include <iostream>
#include "../../include/ast/AbstractExpr.h"
#include "Literal.h"
#include "LiteralInt.h"
#include "LiteralBool.h"
#include "LiteralString.h"

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

LiteralInt *AbstractExpr::createParam(int i) {
    return new LiteralInt(i);
}

LiteralBool *AbstractExpr::createParam(bool b) {
    return new LiteralBool(b);
}

LiteralString *AbstractExpr::createParam(const char *str) {
    return new LiteralString(str);
}

AbstractExpr *AbstractExpr::createParam(AbstractExpr *abstractExpr) {
    return abstractExpr;
}

void to_json(json &j, const AbstractExpr &absExpr) {
    j = absExpr.toJson();
}