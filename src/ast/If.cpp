#include "If.h"
#include "LiteralBool.h"

json If::toJson() const {
  json j;
  j["type"] = getNodeName();
  j["condition"] = this->condition->toJson();
  if (thenBranch != nullptr) j["thenBranch"] = this->thenBranch->toJson();
  if (elseBranch != nullptr) j["elseBranch"] = this->elseBranch->toJson();
  return j;
}

If::If(AbstractExpr *condition, AbstractStatement *thenBranch, AbstractStatement *elseBranch)
    : condition(condition), thenBranch(thenBranch), elseBranch(elseBranch) {}

If::If(AbstractExpr *condition, AbstractStatement *thenBranch) : condition(condition), thenBranch(thenBranch) {
  elseBranch = nullptr;
}

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

AbstractNode *If::createClonedNode(bool keepOriginalUniqueNodeId) {
  return new If(this->condition->cloneRecursiveDeep(keepOriginalUniqueNodeId)->castTo<AbstractExpr>(),
                this->thenBranch->cloneRecursiveDeep(keepOriginalUniqueNodeId)->castTo<AbstractStatement>(),
                this->elseBranch->cloneRecursiveDeep(keepOriginalUniqueNodeId)->castTo<AbstractStatement>());
}
