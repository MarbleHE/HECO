#include "ast_opt/ast/UnaryExpr.h"

json UnaryExpr::toJson() const {
  json j;
  j["type"] = getNodeType();
  j["operator"] = getOperator() ? getOperator()->getOperatorString() : "";
  j["rightOperand"] = getRight() ? getRight()->toJson() : "";
  return j;
}

UnaryExpr::UnaryExpr(UnaryOp op, AbstractExpr *right) {
  setAttributes(op, right);
}

void UnaryExpr::accept(Visitor &v) {
  v.visit(*this);
}

Operator *UnaryExpr::getOperator() const {
  return dynamic_cast<Operator *>(getChildAtIndex(0, true));
}

AbstractExpr *UnaryExpr::getRight() const {
  return dynamic_cast<AbstractExpr *>(getChildAtIndex(1, true));
}

std::string UnaryExpr::getNodeType() const {
  return "UnaryExpr";
}

UnaryExpr::~UnaryExpr() {
  for (auto &child : getChildren()) delete child;
}

bool UnaryExpr::supportsCircuitMode() {
  return true;
}

int UnaryExpr::getMaxNumberChildren() {
  return 2;
}

void UnaryExpr::setAttributes(UnaryOp op, AbstractExpr *expr) {
  removeChildren();
  auto nodesToBeAdded = std::vector<AbstractNode *>({new Operator(op), expr});
  addChildren(nodesToBeAdded, true);
}

UnaryExpr *UnaryExpr::clone(bool keepOriginalUniqueNodeId) {
  try {
    auto clonedNode = new UnaryExpr(std::get<UnaryOp>(this->getOperator()->getOperatorSymbol()),
                                    this->getRight()->clone(keepOriginalUniqueNodeId)->castTo<AbstractExpr>());
    clonedNode->updateClone(keepOriginalUniqueNodeId, this);
    return clonedNode;
  } catch (std::bad_variant_access &exc) {
    throw std::runtime_error(
        "Failed to clone UnaryExpr - unexpected Operator encountered! Expected operator of Enum UnaryOp.");
  }
}

bool UnaryExpr::isEqual(AbstractExpr *other) {
  if (auto otherAsUExp = dynamic_cast<UnaryExpr *>(other)) {
    auto sameOperator = this->getOperator()->equals(otherAsUExp->getOperator()->getOperatorSymbol());
    auto sameValue = this->getRight()->isEqual(otherAsUExp->getRight());
    return sameOperator && sameValue;
  }
  return false;
}
std::string UnaryExpr::toString(bool printChildren) const {
  return AbstractNode::generateOutputString(printChildren, {});
}
