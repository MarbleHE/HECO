#include "../../include/ast/If.h"


json If::toJson() const {
    json j;
    j["type"] = "If";
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

