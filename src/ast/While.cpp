#include <exception>
#include "While.h"
#include "LiteralBool.h"

While::While(AbstractExpr *condition, AbstractStatement *body) {
  setAttributes(condition, body);
}

json While::toJson() const {
  json j;
  j["type"] = getNodeType();
  j["condition"] = getCondition()->toJson();
  j["body"] = getBody()->toJson();
  return j;
}

void While::accept(Visitor &v) {
  v.visit(*this);
}

AbstractExpr *While::getCondition() const {
  return reinterpret_cast<AbstractExpr *>(getChildAtIndex(0));
}

AbstractStatement *While::getBody() const {
  return reinterpret_cast<AbstractStatement *>(getChildAtIndex(1));
}

std::string While::getNodeType() const {
  return "While";
}

While *While::clone(bool keepOriginalUniqueNodeId) {
  auto clonedWhile = new While(getCondition()->clone(keepOriginalUniqueNodeId)->castTo<AbstractExpr>(),
                               getBody()->clone(false)->castTo<AbstractStatement>());
  if (keepOriginalUniqueNodeId) clonedWhile->setUniqueNodeId(this->getUniqueNodeId());
  return clonedWhile;
}

void While::setAttributes(AbstractExpr *loopCondition, AbstractStatement *loopBody) {
  removeChildren();
  addChildren({loopCondition, loopBody}, true);
}

int While::getMaxNumberChildren() {
  return 2;
}

bool While::supportsCircuitMode() {
  return true;
}
