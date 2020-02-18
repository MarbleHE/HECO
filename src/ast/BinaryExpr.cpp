#include <vector>
#include "BinaryExpr.h"
#include "Variable.h"

json BinaryExpr::toJson() const {
  json j;
  j["type"] = getNodeName();
  j["leftOperand"] = getLeft() ? getLeft()->toJson() : "";
  j["operator"] = getOp() ? getOp()->getOperatorString() : "";
  j["rightOperand"] = getRight() ? getRight()->toJson() : "";
  return j;
}

BinaryExpr::BinaryExpr(AbstractExpr *left, OpSymb::BinaryOp op, AbstractExpr *right) {
  setAttributes(left, new Operator(op), right);
}

BinaryExpr::BinaryExpr(OpSymb::BinaryOp op) {
  setAttributes(nullptr, new Operator(op), nullptr);
}

BinaryExpr::BinaryExpr() {
  setAttributes(nullptr, nullptr, nullptr);
}

AbstractExpr *BinaryExpr::getLeft() const {
  return reinterpret_cast<AbstractExpr * >(getChildAtIndex(0, true));
}

Operator *BinaryExpr::getOp() const {
  return reinterpret_cast<Operator *>(getChildAtIndex(1, true));
}

AbstractExpr *BinaryExpr::getRight() const {
  return reinterpret_cast<AbstractExpr * >(getChildAtIndex(2, true));
}

void BinaryExpr::accept(Visitor &v) {
  v.visit(*this);
}

std::string BinaryExpr::getNodeName() const {
  return "BinaryExpr";
}

BinaryExpr *BinaryExpr::contains(BinaryExpr *bexpTemplate, AbstractExpr *excludedSubtree) {
  if (excludedSubtree != nullptr && this == excludedSubtree) {
    return nullptr;
  } else {
    bool emptyOrEqualLeft = (!bexpTemplate->getLeft() || bexpTemplate->getLeft() == this->getLeft());
    bool emptyOrEqualRight = (!bexpTemplate->getRight() || bexpTemplate->getRight() == this->getRight());
    bool emptyOrEqualOp = (bexpTemplate->getOp()->isUndefined() || *this->getOp() == *bexpTemplate->getOp());
    return (emptyOrEqualLeft && emptyOrEqualRight && emptyOrEqualOp) ? this : nullptr;
  }
}

void BinaryExpr::setAttributes(AbstractExpr *leftOperand, Operator *operatore, AbstractExpr *rightOperand) {
  // update tree structure
  removeChildren();
  addChildren({leftOperand, operatore, rightOperand}, false);
  AbstractNode::addParentTo(this, {leftOperand, operatore, rightOperand});
}

void BinaryExpr::swapOperandsLeftAWithRightB(BinaryExpr *bexpA, BinaryExpr *bexpB) {
  auto lopA = bexpA->getLeft();
  auto ropB = bexpB->getRight();
  bexpA->setAttributes(ropB, bexpA->getOp(), bexpA->getRight());
  bexpB->setAttributes(bexpB->getLeft(), bexpB->getOp(), lopA);
}

BinaryExpr::~BinaryExpr() = default;

bool BinaryExpr::contains(Variable *var) {
  return (getLeft()->contains(var) || getRight()->contains(var));
}

bool BinaryExpr::isEqual(AbstractExpr *other) {
  if (auto otherBexp = dynamic_cast<BinaryExpr *>(other)) {
    auto sameLeft = this->getLeft()->isEqual(otherBexp->getLeft());
    auto sameRight = this->getRight()->isEqual(otherBexp->getRight());
    auto sameOp = *this->getOp() == *otherBexp->getOp();
    return sameLeft && sameRight && sameOp;
  }
  return false;
}

int BinaryExpr::countByTemplate(AbstractExpr *abstractExpr) {
  // check if abstractExpr is of type BinaryExpr
  if (auto expr = dynamic_cast<BinaryExpr *>(abstractExpr)) {
    // check if current BinaryExpr fulfills requirements of template abstractExpr
    // also check left and right operands for nested BinaryExps
    return (this->contains(expr, nullptr) != nullptr ? 1 : 0)
           + getLeft()->countByTemplate(abstractExpr)
           + getRight()->countByTemplate(abstractExpr);
  }
  return 0;
}

std::vector<std::string> BinaryExpr::getVariableIdentifiers() {
  auto leftVec = getLeft()->getVariableIdentifiers();
  auto rightVec = getRight()->getVariableIdentifiers();
  leftVec.reserve(leftVec.size() + rightVec.size());
  leftVec.insert(leftVec.end(), rightVec.begin(), rightVec.end());
  return leftVec;
}

int BinaryExpr::getMaxNumberChildren() {
  return 3;
}

bool BinaryExpr::supportsCircuitMode() {
  return true;
}

AbstractNode *BinaryExpr::createClonedNode(bool keepOriginalUniqueNodeId) {
  return new BinaryExpr(this->getLeft()->cloneRecursiveDeep(keepOriginalUniqueNodeId)->castTo<AbstractExpr>(),
                        this->getOp()->cloneRecursiveDeep(keepOriginalUniqueNodeId)->castTo<Operator>(),
                        this->getRight()->cloneRecursiveDeep(keepOriginalUniqueNodeId)->castTo<AbstractExpr>());
}
