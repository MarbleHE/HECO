#include "../../include/ast/If.h"

json If::toJson() const {
    json j;
    j["type"] = getNodeName();
    j["condition"] = this->condition->toJson();
    j["thenBranch"] = this->thenBranch->toJson();
    j["elseBranch"] = this->elseBranch->toJson();
    return j;
}

If::If(AbstractExpr *condition, AbstractStatement *thenBranch, AbstractStatement *elseBranch) : condition(condition),
                                                                                                thenBranch(thenBranch),
                                                                                                elseBranch(
                                                                                                        elseBranch) {}

If::If(AbstractExpr *condition, AbstractStatement *thenBranch) : condition(condition), thenBranch(thenBranch) {}

void If::accept(Visitor &v) {
    v.visit(*this);
}

std::string If::getNodeName() const {
    return "If";
}

AbstractExpr *If::getCondition() const {
    return condition;
}

AbstractStatement *If::getThenBranch() const {
    return thenBranch;
}

AbstractStatement *If::getElseBranch() const {
    return elseBranch;
}

If::~If() {
    delete condition;
    delete thenBranch;
    delete elseBranch;
}
