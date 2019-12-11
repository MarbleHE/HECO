#include "../../include/ast/If.h"

If::If(std::unique_ptr<AbstractExpr> condition, std::unique_ptr<AbstractStatement> thenBranch)
        : condition(std::move(condition)), thenBranch(std::move(thenBranch)) {}

If::If(std::unique_ptr<AbstractExpr> condition, std::unique_ptr<AbstractStatement> thenBranch,
       std::unique_ptr<AbstractStatement> elseBranch)
        : condition(std::move(condition)), thenBranch(std::move(thenBranch)), elseBranch(std::move(elseBranch)) {}

json If::toJson() const {
    json j;
    j["type"] = "If";
    j["condition"] = this->condition->toJson();
    j["thenBranch"] = this->thenBranch->toJson();
    j["elseBranch"] = this->elseBranch->toJson();
    return j;
}

