#include "ast_opt/ast/If.h"
#include "ast_opt/ast/LiteralBool.h"

json If::toJson() const {
  json j;
  j["type"] = getNodeType();
  j["condition"] = getCondition()->toJson();
  if (getThenBranch()!=nullptr) j["thenBranch"] = getThenBranch()->toJson();
  if (getElseBranch()!=nullptr) j["elseBranch"] = getElseBranch()->toJson();
  return j;
}

If::If(AbstractExpr *condition, AbstractStatement *thenBranch, AbstractStatement *elseBranch) {
  setAttributes(condition, thenBranch, elseBranch);
}

If::If(AbstractExpr *condition, AbstractStatement *thenBranch) {
  setAttributes(condition, thenBranch, nullptr);
}

void If::accept(Visitor &v) {
  v.visit(*this);
}

std::string If::getNodeType() const {
  return "If";
}

AbstractExpr *If::getCondition() const {
  return getChildAtIndex(0)->castTo<AbstractExpr>();
}

AbstractStatement *If::getThenBranch() const {
  return getChildAtIndex(1) ? getChildAtIndex(1)->castTo<AbstractStatement>() : nullptr;
}

AbstractStatement *If::getElseBranch() const {
  // make sure that there exists an Else branch
  if (getChildAtIndex(2)!=nullptr) return getChildAtIndex(2)->castTo<AbstractStatement>();
  return nullptr;
}

If::~If() = default;

If *If::clone(bool keepOriginalUniqueNodeId) {
  auto clonedNode = new If(getCondition()->clone(keepOriginalUniqueNodeId)->castTo<AbstractExpr>(),
                           getThenBranch()->clone(keepOriginalUniqueNodeId)->castTo<AbstractStatement>(),
                           getElseBranch()->clone(keepOriginalUniqueNodeId)->castTo<AbstractStatement>());
  clonedNode->updateClone(keepOriginalUniqueNodeId, this);
  return clonedNode;
}
int If::getMaxNumberChildren() {
  return 3;
}
bool If::supportsCircuitMode() {
  return true;
}

void If::setAttributes(AbstractExpr *condition, AbstractStatement *thenBranch, AbstractStatement *elseBranch) {
  // update tree structure
  removeChildren();
  addChildren({condition, thenBranch, elseBranch}, true);
}
std::string If::toString(bool printChildren) const {
  return AbstractNode::generateOutputString(printChildren, {});
}
