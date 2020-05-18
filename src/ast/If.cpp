#include "ast_opt/ast/If.h"
#include "ast_opt/ast/LiteralBool.h"
#include "ast_opt/ast/Block.h"

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

Block *If::getThenBranch() const {
  return getChildAtIndex(1) ? getChildAtIndex(1)->castTo<Block>() : nullptr;
}

Block *If::getElseBranch() const {
  return getChildAtIndex(2) ? getChildAtIndex(2)->castTo<Block>() : nullptr;
}

If::~If() = default;

If *If::clone(bool keepOriginalUniqueNodeId) const {
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

  addChild(condition, true);

  auto thenBranchAsBlock = dynamic_cast<Block *>(thenBranch);
  if (!thenBranchAsBlock) {
    thenBranchAsBlock = new Block;
    thenBranchAsBlock->addChild(thenBranch, true);
  }
  addChild(thenBranchAsBlock, true);

  auto elseBranchAsBlock = dynamic_cast<Block *>(elseBranch);
  if (!elseBranchAsBlock) {
    elseBranchAsBlock = new Block;
    elseBranchAsBlock->addChild(elseBranch, true);
  }
  addChild(elseBranchAsBlock, true);

}
std::string If::toString(bool printChildren) const {
  return AbstractNode::generateOutputString(printChildren, {});
}
