#include "../../include/ast/BinaryExpr.h"
#include "../../include/ast/Variable.h"

json BinaryExpr::toJson() const {
    json j;
    j["type"] = getNodeName();
    j["leftOperand"] = this->left->toJson();
    j["operator"] = this->op->getOperatorString();
    j["rightOperand"] = this->right->toJson();
    return j;
}

BinaryExpr::BinaryExpr(AbstractExpr *left, OpSymb::BinaryOp op, AbstractExpr *right) : left(left), right(right) {
    this->op = new Operator(op);
}

AbstractExpr *BinaryExpr::getLeft() const {
    return left;
}

Operator &BinaryExpr::getOp() const {
    return *op;
}

AbstractExpr *BinaryExpr::getRight() const {
    return right;
}


void BinaryExpr::accept(Visitor &v) {
    v.visit(*this);
}

std::string BinaryExpr::getNodeName() const {
    return "BinaryExpr";
}

BinaryExpr::BinaryExpr(Operator *op) : op(op) {
    this->left = nullptr;
    this->right = nullptr;
}

BinaryExpr *BinaryExpr::containsValuesFrom(BinaryExpr *bexpTemplate) {
    if ((bexpTemplate->getLeft() == nullptr || bexpTemplate->getLeft() == this->getLeft())
        && (bexpTemplate->getRight() == nullptr || bexpTemplate->getRight() == this->getRight())
        && (bexpTemplate->getOp().getOperatorString().empty() ||
            bexpTemplate->getOp().getOperatorString() == this->getOp().getOperatorString())) {
        return this;
    }
    return nullptr;
}

void BinaryExpr::setLeft(AbstractExpr *left) {
    BinaryExpr::left = left;
}

void BinaryExpr::setOp(Operator *op) {
    BinaryExpr::op = op;
}

void BinaryExpr::setRight(AbstractExpr *right) {
    BinaryExpr::right = right;
}

void BinaryExpr::swapOperandsLeftAWithRightB(BinaryExpr *bexpA, BinaryExpr *bexpB) {
    auto previousBexpLeftOp = bexpA->getLeft();
    bexpA->setLeft(bexpB->getRight());
    bexpB->setRight(previousBexpLeftOp);
}

BinaryExpr::~BinaryExpr() {
    delete left;
    delete right;
    delete op;
}


