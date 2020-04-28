#include "For.h"
#include "AbstractExpr.h"
#include "Block.h"

std::string For::getNodeType() const {
  return "For";
}

void For::accept(Visitor &v) {
  v.visit(*this);
}

For::For(AbstractStatement *initializer,
         AbstractExpr *condition,
         AbstractStatement *update,
         AbstractStatement *statementToBeExecuted) {
  setAttributes(initializer, condition, update, statementToBeExecuted);
}

AbstractStatement *For::getInitializer() const {
  return reinterpret_cast<AbstractStatement *>(getChildAtIndex(0));
}

AbstractExpr *For::getCondition() const {
  return reinterpret_cast<AbstractExpr *>(getChildAtIndex(1));
}

AbstractStatement *For::getUpdateStatement() const {
  return reinterpret_cast<AbstractStatement *>(getChildAtIndex(2));
}

AbstractStatement *For::getStatementToBeExecuted() const {
  return reinterpret_cast<AbstractStatement *>(getChildAtIndex(3));
}

void For::setAttributes(AbstractStatement *initializer,
                        AbstractExpr *condition,
                        AbstractStatement *update,
                        AbstractStatement *statementToBeExecuted) {
  removeChildren();
  if (dynamic_cast<Block *>(statementToBeExecuted)==nullptr) {
    statementToBeExecuted = new Block(statementToBeExecuted);
  }
  addChildren({initializer, condition, update, statementToBeExecuted}, true);
}

int For::getMaxNumberChildren() {
  return 4;
}

AbstractNode *For::clone(bool keepOriginalUniqueNodeId) {

  auto clonedInitializer = (getInitializer()==nullptr)
                           ? nullptr
                           : getInitializer()->clone(keepOriginalUniqueNodeId)->castTo<AbstractStatement>();
  auto clonedCondition = (getCondition()==nullptr)
                         ? nullptr
                         : getCondition()->clone(keepOriginalUniqueNodeId)->castTo<AbstractExpr>();
  auto clonedUpdater = (getUpdateStatement()==nullptr)
                       ? nullptr
                       : getUpdateStatement()->clone(false)->castTo<AbstractStatement>();
  auto clonedBody = (getStatementToBeExecuted()==nullptr)
                    ? nullptr
                    : getStatementToBeExecuted()->clone(keepOriginalUniqueNodeId)->castTo<AbstractStatement>();

  auto clonedNode = new For(clonedInitializer, clonedCondition, clonedUpdater, clonedBody);
  clonedNode->updateClone(keepOriginalUniqueNodeId, this);
  return clonedNode;
}

std::string For::toString(bool printChildren) const {
  return AbstractNode::generateOutputString(printChildren, {});
}

bool For::supportsCircuitMode() {
  return true;
}

json For::toJson() const {
  json j;
  j["type"] = getNodeType();
  j["condition"] = getCondition()->toJson();
  j["initializer"] = getInitializer()->toJson();
  j["update"] = getUpdateStatement()->toJson();
  j["statement"] = getStatementToBeExecuted()->toJson();
  return j;
}

bool For::isEqual(AbstractStatement *other) {
  if (auto otherFor = dynamic_cast<For *>(other)) {
    auto sameInitializer = (getInitializer()==nullptr && otherFor->getInitializer()==nullptr)
        || getInitializer()->isEqual(otherFor->getInitializer());
    auto sameCondition = (getCondition()==nullptr && otherFor->getCondition()==nullptr)
        || getCondition()->isEqual(otherFor->getCondition());
    auto sameUpdateStmt = (getUpdateStatement()==nullptr && otherFor->getUpdateStatement()==nullptr)
        || getUpdateStatement()->isEqual(otherFor->getUpdateStatement());
    auto sameBody = getStatementToBeExecuted()->isEqual(otherFor->getStatementToBeExecuted());
    return sameInitializer && sameCondition && sameUpdateStmt && sameBody;
  }
  return false;
}
