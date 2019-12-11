#include "../../include/ast/BinaryExpr.h"

json BinaryExpr::toJson() const {
    json j;
    j["type"] = "BinaryExpr";
    j["leftOperand"] = this->left->toJson();
    j["operator"] = this->op;
    j["rightOperand"] = this->right->toJson();
    return j;
}

AbstractExpr *BinaryExpr::getLeft() const {
    return left.get();
}

BinaryOperator BinaryExpr::getOp() const {
    return op;
}

AbstractExpr *BinaryExpr::getRight() const {
    return right.get();
}

BinaryExpr::BinaryExpr(std::unique_ptr<AbstractExpr> left, const BinaryOperator &op,
                       std::unique_ptr<AbstractExpr> right) : left(std::move(left)), right(std::move(right)), op(op) {

}
