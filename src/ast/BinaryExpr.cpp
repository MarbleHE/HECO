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

BinaryExpr::BinaryExpr(OpSymb::BinaryOp op) : op(new Operator(op)) {
    this->left = nullptr;
    this->right = nullptr;
}

BinaryExpr *BinaryExpr::contains(BinaryExpr *bexpTemplate, AbstractExpr *excludedSubtree) {
    if (excludedSubtree != nullptr && this == excludedSubtree) {
        return nullptr;
    } else {
        bool emptyOrEqualLeft = (!bexpTemplate->getLeft() || bexpTemplate->getLeft() == this->getLeft());
        bool emptyOrEqualRight = (!bexpTemplate->getRight() || bexpTemplate->getRight() == this->getRight());
        bool emptyOrEqualOp = (bexpTemplate->getOp().isUndefined() || this->getOp() == bexpTemplate->getOp());
        return (emptyOrEqualLeft && emptyOrEqualRight && emptyOrEqualOp) ? this : nullptr;
    }
}

void BinaryExpr::setLeft(AbstractExpr *value) {
    this->left = value;
}

void BinaryExpr::setOp(Operator *operatore) {
    this->op = operatore;
}

void BinaryExpr::setRight(AbstractExpr *rhs) {
    this->right = rhs;
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

bool BinaryExpr::contains(Variable *var) {
    return (getLeft()->contains(var) || getRight()->contains(var));
}

bool BinaryExpr::isEqual(AbstractExpr *other) {
    if (auto otherBexp = dynamic_cast<BinaryExpr *>(other)) {
        auto sameLeft = this->getLeft()->isEqual(otherBexp->getLeft());
        auto sameRight = this->getRight()->isEqual(otherBexp->getRight());
        auto sameOp = this->getOp() == otherBexp->getOp();
        return sameLeft && sameRight && sameOp;
    }
    return false;
}


