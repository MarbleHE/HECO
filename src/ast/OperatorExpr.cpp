#include "OperatorExpr.h"
#include <utility>
#include "Operator.h"

int OperatorExpr::getMaxNumberChildren() {
  return -1;
}

bool OperatorExpr::supportsCircuitMode() {
  return true;
}

OperatorExpr::OperatorExpr(Operator *op) {
  setAttributes(op, {});
}

OperatorExpr::OperatorExpr(Operator *op, std::vector<AbstractExpr *> operands) {
  setAttributes(op, std::move(operands));
}

OperatorExpr::OperatorExpr(AbstractExpr *lhsOperand, Operator *op, AbstractExpr *rhsOperand) {
  setAttributes(op, {lhsOperand, rhsOperand});
}

std::string OperatorExpr::toString(bool printChildren) const {
  return AbstractNode::generateOutputString(printChildren, {});
}

OperatorExpr *OperatorExpr::clone(bool keepOriginalUniqueNodeId) {
  // clone operator
  auto clonedOperator = getOperator()->clone(keepOriginalUniqueNodeId);
  // clone all operands
  std::vector<AbstractExpr *> clonedAes;
  std::transform(children.begin(), children.end(), std::back_inserter(clonedAes),
                 [keepOriginalUniqueNodeId](AbstractNode *node) -> AbstractExpr * {
                   return node->clone(keepOriginalUniqueNodeId)->castTo<AbstractExpr>();
                 });
  auto clonedOperatorExpr = new OperatorExpr(clonedOperator, clonedAes);
  clonedOperatorExpr->updateClone(keepOriginalUniqueNodeId, this);
  return clonedOperatorExpr;
}

void OperatorExpr::accept(Visitor &v) {
  v.visit(*this);
}

std::string OperatorExpr::getNodeType() const {
  return std::string("OperatorExpr");
}

void OperatorExpr::addOperand(AbstractExpr *operand) {
  // simply append the new operand to the end of the operands list
  addChild(operand, true);
}

void OperatorExpr::setOperator(Operator *op) {
  // child at index 0 is always the operator
  auto curOperator = getChildAtIndex(0);
  replaceChild(curOperator, op);
  delete curOperator;
}

OperatorExpr::OperatorExpr() = default;

void OperatorExpr::setAttributes(Operator *op, std::vector<AbstractExpr *> abstractExprs) {
  removeChildren();
  addChild(op);
  std::vector<AbstractNode *> abstractExprsVec(abstractExprs.begin(), abstractExprs.end());
  addChildren(abstractExprsVec, true);
}

bool OperatorExpr::isLogicalExpr() const {
  return getOperator()->isLogCompOp();
}

Operator *OperatorExpr::getOperator() const {
  return reinterpret_cast<Operator *>(getChildAtIndex(0));
}

bool OperatorExpr::isArithmeticExpr() const {
  return getOperator()->isArithmeticOp();
}

bool OperatorExpr::isUnaryExpr() const {
  return getOperator()->isUnaryOp();
}

bool OperatorExpr::isEqual(AbstractExpr *other) {
  if (auto expr = dynamic_cast<OperatorExpr *>(other)) {
    if (this->getChildren().size()!=other->getChildren().size()) return false;
    if (!this->getOperator()->equals(expr->getOperator()->getOperatorSymbol())) return false;
    for (unsigned int i = 0; i < getOperands().size(); ++i) {
      if (!getOperands().at(i)->isEqual(expr->getOperands().at(i))) return false;
    }
    return true;
  }
  return false;
}

std::vector<AbstractExpr *> OperatorExpr::getOperands() const {
  std::vector<AbstractExpr *> operands;
  // ++children.begin() because operands start from index 1
  std::transform(++children.begin(), children.end(), std::back_inserter(operands),
                 [](AbstractNode *node) -> AbstractExpr * {
                   return node->castTo<AbstractExpr>();
                 });
  return operands;
}

AbstractExpr *OperatorExpr::getRight() const {
  if (getOperands().size() > 2) {
    throw std::logic_error("OperatorExpr::getRight() only supported for expressions with two operands!");
  }
  return reinterpret_cast<AbstractExpr *>(getChildAtIndex(2));
}

AbstractExpr *OperatorExpr::getLeft() const {
  if (getOperands().size() > 2) {
    throw std::logic_error("OperatorExpr::getLeft() only supported for expressions with two operands!");
  }
  return reinterpret_cast<AbstractExpr *>(getChildAtIndex(1));
}
