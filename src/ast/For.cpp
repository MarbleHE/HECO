#include "ast_opt/ast/For.h"
#include "ast_opt/ast/AbstractExpr.h"
#include "ast_opt/ast/Block.h"

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

Block * For::getInitializer() const {
  return dynamic_cast<Block *>(getChildAtIndex(0));
}

AbstractExpr *For::getCondition() const {
  return dynamic_cast<AbstractExpr *>(getChildAtIndex(1));
}

Block * For::getUpdate() const {
  return dynamic_cast<Block *>(getChildAtIndex(2));
}

Block * For::getBody() const {
  return dynamic_cast<Block *>(getChildAtIndex(3));
}

void For::setAttributes(AbstractStatement *initializer,
                        AbstractExpr *condition,
                        AbstractStatement *update,
                        AbstractStatement *body) {
  removeChildren();
  if (dynamic_cast<Block *>(initializer)==nullptr) {
    initializer = new Block(initializer);
  }
  if (dynamic_cast<Block *>(update)==nullptr) {
    update = new Block(update);
  }
  if (dynamic_cast<Block *>(body)==nullptr) {
    body = new Block(body);
  }
  addChildren({initializer, condition, update, body}, true);
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
  auto clonedUpdater = (getUpdate()==nullptr)
                       ? nullptr
                       : getUpdate()->clone(false)->castTo<AbstractStatement>();
  auto clonedBody = (getBody()==nullptr)
                    ? nullptr
                    : getBody()->clone(keepOriginalUniqueNodeId)->castTo<AbstractStatement>();

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
  j["update"] = getUpdate()->toJson();
  j["statement"] = getBody()->toJson();
  return j;
}

bool For::isEqual(AbstractStatement *other) {
  if (auto otherFor = dynamic_cast<For *>(other)) {
    auto sameInitializer = (getInitializer()==nullptr && otherFor->getInitializer()==nullptr)
        || getInitializer()->isEqual(otherFor->getInitializer());
    auto sameCondition = (getCondition()==nullptr && otherFor->getCondition()==nullptr)
        || getCondition()->isEqual(otherFor->getCondition());
    auto sameUpdateStmt = (getUpdate()==nullptr && otherFor->getUpdate()==nullptr)
        || getUpdate()->isEqual(otherFor->getUpdate());
    auto sameBody = getBody()->isEqual(otherFor->getBody());
    return sameInitializer && sameCondition && sameUpdateStmt && sameBody;
  }
  return false;
}
