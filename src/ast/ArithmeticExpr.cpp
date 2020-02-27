#include <vector>
#include "ArithmeticExpr.h"
#include "Variable.h"

json ArithmeticExpr::toJson() const {
  json j;
  j["type"] = getNodeName();
  j["leftOperand"] = getLeft() ? getLeft()->toJson() : "";
  j["operator"] = getOp() ? getOp()->getOperatorString() : "";
  j["rightOperand"] = getRight() ? getRight()->toJson() : "";
  return j;
}

ArithmeticExpr::ArithmeticExpr(AbstractExpr *left, OpSymb::ArithmeticOp op, AbstractExpr *right) {
  setAttributes(left, new Operator(op), right);
}

ArithmeticExpr::ArithmeticExpr(OpSymb::ArithmeticOp op) {
  setAttributes(nullptr, new Operator(op), nullptr);
}

ArithmeticExpr::ArithmeticExpr() {
  setAttributes(nullptr, nullptr, nullptr);
}

AbstractExpr *ArithmeticExpr::getLeft() const {
  return reinterpret_cast<AbstractExpr * >(getChildAtIndex(0, true));
}

Operator *ArithmeticExpr::getOp() const {
  return reinterpret_cast<Operator *>(getChildAtIndex(1, true));
}

AbstractExpr *ArithmeticExpr::getRight() const {
  return reinterpret_cast<AbstractExpr * >(getChildAtIndex(2, true));
}

void ArithmeticExpr::accept(Visitor &v) {
  v.visit(*this);
}

std::string ArithmeticExpr::getNodeName() const {
  return "ArithmeticExpr";
}

ArithmeticExpr *ArithmeticExpr::contains(ArithmeticExpr *aexpTemplate, AbstractExpr *excludedSubtree) {
  if (excludedSubtree!=nullptr && this==excludedSubtree) {
    return nullptr;
  } else {
    bool emptyOrEqualLeft = (!aexpTemplate->getLeft() || aexpTemplate->getLeft()==this->getLeft());
    bool emptyOrEqualRight = (!aexpTemplate->getRight() || aexpTemplate->getRight()==this->getRight());
    bool emptyOrEqualOp = (aexpTemplate->getOp()->isUndefined() || *this->getOp()==*aexpTemplate->getOp());
    return (emptyOrEqualLeft && emptyOrEqualRight && emptyOrEqualOp) ? this : nullptr;
  }
}

void ArithmeticExpr::setAttributes(AbstractExpr *leftOperand, Operator *operatore, AbstractExpr *rightOperand) {
  removeChildren();
  addChildren({leftOperand, operatore, rightOperand}, true);
}

void ArithmeticExpr::swapOperandsLeftAWithRightB(ArithmeticExpr *aexpA, ArithmeticExpr *aexpB) {
  auto lopA = aexpA->getLeft();
  auto ropB = aexpB->getRight();
  aexpA->setAttributes(ropB, aexpA->getOp(), aexpA->getRight());
  aexpB->setAttributes(aexpB->getLeft(), aexpB->getOp(), lopA);
}

ArithmeticExpr::~ArithmeticExpr() = default;

bool ArithmeticExpr::contains(Variable *var) {
  return (getLeft()->contains(var) || getRight()->contains(var));
}

bool ArithmeticExpr::isEqual(AbstractExpr *other) {
  if (auto otherAexp = dynamic_cast<ArithmeticExpr *>(other)) {
    auto sameLeft = this->getLeft()->isEqual(otherAexp->getLeft());
    auto sameRight = this->getRight()->isEqual(otherAexp->getRight());
    auto sameOp = *this->getOp()==*otherAexp->getOp();
    return sameLeft && sameRight && sameOp;
  }
  return false;
}

int ArithmeticExpr::countByTemplate(AbstractExpr *abstractExpr) {
  // check if abstractExpr is of type ArithmeticExpr
  if (auto expr = dynamic_cast<ArithmeticExpr *>(abstractExpr)) {
    // check if current ArithmeticExpr fulfills requirements of template abstractExpr
    // also check left and right operands for nested arithmetic expressions
    return (this->contains(expr, nullptr)!=nullptr ? 1 : 0)
        + getLeft()->countByTemplate(abstractExpr)
        + getRight()->countByTemplate(abstractExpr);
  }
  return 0;
}

std::vector<std::string> ArithmeticExpr::getVariableIdentifiers() {
  auto leftVec = getLeft()->getVariableIdentifiers();
  auto rightVec = getRight()->getVariableIdentifiers();
  leftVec.reserve(leftVec.size() + rightVec.size());
  leftVec.insert(leftVec.end(), rightVec.begin(), rightVec.end());
  return leftVec;
}

int ArithmeticExpr::getMaxNumberChildren() {
  return 3;
}

bool ArithmeticExpr::supportsCircuitMode() {
  return true;
}

ArithmeticExpr *ArithmeticExpr::clone(bool keepOriginalUniqueNodeId) {
  auto clonedNode = new ArithmeticExpr(this->getLeft()->clone(keepOriginalUniqueNodeId)->castTo<AbstractExpr>(),
                                       this->getOp()->clone(keepOriginalUniqueNodeId)->castTo<Operator>(),
                                       this->getRight()->clone(keepOriginalUniqueNodeId)->castTo<AbstractExpr>());
  if (keepOriginalUniqueNodeId) clonedNode->setUniqueNodeId(this->getUniqueNodeId());
  if (this->isReversed) clonedNode->swapChildrenParents();
  return clonedNode;
}
