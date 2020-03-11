#include <vector>
#include "ArithmeticExpr.h"
#include "Variable.h"

ArithmeticExpr::ArithmeticExpr(AbstractExpr *left, ArithmeticOp op, AbstractExpr *right) {
  setAttributes(left, new Operator(op), right);
}

ArithmeticExpr::ArithmeticExpr(ArithmeticOp op) {
  setAttributes(nullptr, new Operator(op), nullptr);
}

ArithmeticExpr::ArithmeticExpr() {
  setAttributes(nullptr, nullptr, nullptr);
}

void ArithmeticExpr::accept(Visitor &v) {
  v.visit(*this);
}

std::string ArithmeticExpr::getNodeType() const {
  return "ArithmeticExpr";
}

ArithmeticExpr::~ArithmeticExpr() = default;

ArithmeticExpr *ArithmeticExpr::clone(bool keepOriginalUniqueNodeId) {
  auto clonedNode = new ArithmeticExpr(this->getLeft()->clone(keepOriginalUniqueNodeId)->castTo<AbstractExpr>(),
                                       this->getOp()->clone(keepOriginalUniqueNodeId)->castTo<Operator>(),
                                       this->getRight()->clone(keepOriginalUniqueNodeId)->castTo<AbstractExpr>());
  clonedNode->updateClone(keepOriginalUniqueNodeId, this);
  return clonedNode;
}
std::string ArithmeticExpr::toString(bool printChildren) const {
  return AbstractNode::generateOutputString(printChildren, {});
}
