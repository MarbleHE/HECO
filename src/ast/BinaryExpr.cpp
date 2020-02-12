#include "BinaryExpr.h"
#include "Variable.h"

json BinaryExpr::toJson() const {
  json j;
  j["type"] = getNodeName();
  j["leftOperand"] = this->left->toJson();
  j["operator"] = this->op->getOperatorString();
  j["rightOperand"] = this->right->toJson();
  return j;
}

BinaryExpr::BinaryExpr(AbstractExpr* left, OpSymb::BinaryOp op, AbstractExpr* right) {
  setAttributes(left, new Operator(op), right);
}

AbstractExpr* BinaryExpr::getLeft() const {
  return left;
}

Operator &BinaryExpr::getOp() const {
  return *op;
}

AbstractExpr* BinaryExpr::getRight() const {
  return right;
}

void BinaryExpr::accept(Visitor &v) {
  v.visit(*this);
}

std::string BinaryExpr::getNodeName() const {
  return "BinaryExpr";
}

BinaryExpr::BinaryExpr(OpSymb::BinaryOp op) : op(new Operator(op)) {
  this->left = nullptr;
  this->right = nullptr;
}

BinaryExpr* BinaryExpr::contains(BinaryExpr* bexpTemplate, AbstractExpr* excludedSubtree) {
  if (excludedSubtree != nullptr && this == excludedSubtree) {
    return nullptr;
  } else {
    bool emptyOrEqualLeft = (!bexpTemplate->getLeft() || bexpTemplate->getLeft() == this->getLeft());
    bool emptyOrEqualRight = (!bexpTemplate->getRight() || bexpTemplate->getRight() == this->getRight());
    bool emptyOrEqualOp = (bexpTemplate->getOp().isUndefined() || this->getOp() == bexpTemplate->getOp());
    return (emptyOrEqualLeft && emptyOrEqualRight && emptyOrEqualOp) ? this : nullptr;
  }
}

void BinaryExpr::setAttributes(AbstractExpr* leftOperand, Operator* operatore, AbstractExpr* rightOperand) {
  this->left = leftOperand;
  this->op = operatore;
  this->right = rightOperand;
  // update tree structure
  this->removeChildren();
  this->addChildren({leftOperand, operatore, rightOperand});
  Node::addParent(this, {leftOperand, operatore, rightOperand});
}

void BinaryExpr::swapOperandsLeftAWithRightB(BinaryExpr* bexpA, BinaryExpr* bexpB) {
  auto lopA = bexpA->getLeft();
  auto ropB = bexpB->getRight();
  bexpA->setAttributes(ropB, &bexpA->getOp(), bexpA->getRight());
  bexpB->setAttributes(bexpB->getLeft(), &bexpB->getOp(), lopA);
}

BinaryExpr::~BinaryExpr() {
  delete left;
  delete right;
  delete op;
}

bool BinaryExpr::contains(Variable* var) {
  return (getLeft()->contains(var) || getRight()->contains(var));
}

bool BinaryExpr::isEqual(AbstractExpr* other) {
  if (auto otherBexp = dynamic_cast<BinaryExpr*>(other)) {
    auto sameLeft = this->getLeft()->isEqual(otherBexp->getLeft());
    auto sameRight = this->getRight()->isEqual(otherBexp->getRight());
    auto sameOp = this->getOp() == otherBexp->getOp();
    return sameLeft && sameRight && sameOp;
  }
  return false;
}

Literal* BinaryExpr::evaluate(Ast &ast) {
  // we first need to evaluate the left-handside and right-handside as they can consists of nested binary expressions
  return this->getOp().applyOperator(this->getLeft()->evaluate(ast), this->getRight()->evaluate(ast));
}

BinaryExpr::BinaryExpr() : left(nullptr), op(nullptr), right(nullptr) {}

int BinaryExpr::countByTemplate(AbstractExpr* abstractExpr) {
  // check if abstractExpr is of type BinaryExpr
  if (auto expr = dynamic_cast<BinaryExpr*>(abstractExpr)) {
    // check if current BinaryExpr fulfills requirements of template abstractExpr
    // also check left and right operands for nested BinaryExps
    return (this->contains(expr, nullptr) != nullptr ? 1 : 0)
        + left->countByTemplate(abstractExpr)
        + right->countByTemplate(abstractExpr);
  }
  return 0;
}

std::vector<std::string> BinaryExpr::getVariableIdentifiers() {
  auto leftVec = left->getVariableIdentifiers();
  auto rightVec = right->getVariableIdentifiers();
  leftVec.reserve(leftVec.size() + rightVec.size());
  leftVec.insert(leftVec.end(), rightVec.begin(), rightVec.end());
  return leftVec;
}

