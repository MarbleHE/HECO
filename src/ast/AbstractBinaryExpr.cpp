#include "AbstractBinaryExpr.h"
#include "Operator.h"
#include "ArithmeticExpr.h"
#include "LogicalExpr.h"

void AbstractBinaryExpr::setAttributes(AbstractExpr *leftOperand, Operator *operatore, AbstractExpr *rightOperand) {
  removeChildren();
  addChildren({leftOperand, operatore, rightOperand}, true);
}

AbstractExpr *AbstractBinaryExpr::getLeft() const {
  return reinterpret_cast<AbstractExpr * >(getChildAtIndex(0, true));
}

Operator *AbstractBinaryExpr::getOp() const {
  return reinterpret_cast<Operator *>(getChildAtIndex(1, true));
}

AbstractExpr *AbstractBinaryExpr::getRight() const {
  return reinterpret_cast<AbstractExpr * >(getChildAtIndex(2, true));
}

int AbstractBinaryExpr::getMaxNumberChildren() {
  return 3;
}

json AbstractBinaryExpr::toJson() const {
  json j;
  j["type"] = getNodeType();
  j["leftOperand"] = getLeft() ? getLeft()->toJson() : "";
  j["operator"] = getOp() ? getOp()->getOperatorString() : "";
  j["rightOperand"] = getRight() ? getRight()->toJson() : "";
  return j;
}

std::vector<std::string> AbstractBinaryExpr::getVariableIdentifiers() {
  auto leftVec = getLeft()->getVariableIdentifiers();
  auto rightVec = getRight()->getVariableIdentifiers();
  leftVec.reserve(leftVec.size() + rightVec.size());
  leftVec.insert(leftVec.end(), rightVec.begin(), rightVec.end());
  return leftVec;
}

bool AbstractBinaryExpr::isEqual(AbstractExpr *other) {
  if (auto otherLexp = dynamic_cast<AbstractBinaryExpr *>(other)) {
    auto sameLeft = this->getLeft()->isEqual(otherLexp->getLeft());
    auto sameRight = this->getRight()->isEqual(otherLexp->getRight());
    auto sameOp = *this->getOp()==*otherLexp->getOp();
    return sameLeft && sameRight && sameOp;
  }
  return false;
}

bool AbstractBinaryExpr::supportsCircuitMode() {
  return true;
}

bool AbstractBinaryExpr::contains(Variable *var) {
  return (getLeft()->contains(var) || getRight()->contains(var));
}

int AbstractBinaryExpr::countByTemplate(AbstractExpr *abstractExpr) {
  if (auto expr = dynamic_cast<AbstractBinaryExpr *>(abstractExpr)) {
    // check if current AbstractBinaryExpr fulfills requirements of template abstractExpr
    // also check left and right operands which can contain nested arithmetic expressions
    return (this->contains(expr, nullptr)!=nullptr ? 1 : 0)
        + getLeft()->countByTemplate(abstractExpr)
        + getRight()->countByTemplate(abstractExpr);
  } else {
    return 0;
  }
}

AbstractBinaryExpr *AbstractBinaryExpr::contains(AbstractBinaryExpr *aexpTemplate, AbstractExpr *excludedSubtree) {
  if (excludedSubtree!=nullptr && this==excludedSubtree) {
    return nullptr;
  } else {
    bool emptyOrEqualLeft = (!aexpTemplate->getLeft() || aexpTemplate->getLeft()==this->getLeft());
    bool emptyOrEqualRight = (!aexpTemplate->getRight() || aexpTemplate->getRight()==this->getRight());
    bool emptyOrEqualOp = (aexpTemplate->getOp()->isUndefined() || *this->getOp()==*aexpTemplate->getOp());
    return (emptyOrEqualLeft && emptyOrEqualRight && emptyOrEqualOp) ? this : nullptr;
  }
}

void AbstractBinaryExpr::swapOperandsLeftAWithRightB(AbstractBinaryExpr *aexpA, AbstractBinaryExpr *aexpB) {
  auto lopA = aexpA->getLeft();
  auto ropB = aexpB->getRight();
  aexpA->setAttributes(ropB, aexpA->getOp(), aexpA->getRight());
  aexpB->setAttributes(aexpB->getLeft(), aexpB->getOp(), lopA);
}
