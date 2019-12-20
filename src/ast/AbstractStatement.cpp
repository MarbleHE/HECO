#include <iostream>
#include "../../include/ast/AbstractStatement.h"
#include "../../include/ast/Block.h"
#include "../../include/ast/Call.h"
#include "BinaryExpr.h"

std::string AbstractStatement::toString() const {
    return this->toJson().dump();
}

json AbstractStatement::toJson() const {
    return json({"type", "AbstractStatement"});
}

void AbstractStatement::accept(Visitor &v) {
    std::cout << "This shouldn't be executed!" << std::endl;
}

BinaryExpr *AbstractStatement::contains(BinaryExpr *bexpTemplate, BinaryExpr *excludedSubtree) {
    return nullptr;
}

std::string AbstractStatement::getVarTargetIdentifier() {
    return std::string();
}

std::ostream &operator<<(std::ostream &outs, const AbstractStatement &obj) {
    return outs << obj.toString();
}

void to_json(json &j, const AbstractStatement &absStat) {
    j = absStat.toJson();
}


void to_json(json &j, const AbstractStatement *absStat) {
    j = absStat->toJson();
}


