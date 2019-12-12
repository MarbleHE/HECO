#include <LiteralInt.h>
#include "../../include/ast/BinaryExpr.h"
#include "../../include/ast/Variable.h"

json BinaryExpr::toJson() const {
    json j;
    j["type"] = "BinaryExpr";
    j["leftOperand"] = this->left->toJson();
    j["operator"] = this->op;
    j["rightOperand"] = this->right->toJson();
    return j;
}


BinaryExpr::BinaryExpr(AbstractExpr *left, BinaryOperator op, AbstractExpr *right) : left(left), op(op), right(right) {}

AbstractExpr *BinaryExpr::getLeft() const {
    return left;
}

BinaryOperator BinaryExpr::getOp() const {
    return op;
}

AbstractExpr *BinaryExpr::getRight() const {
    return right;
}

BinaryExpr::BinaryExpr(const std::string &variableLeft, BinaryOperator op, const std::string &variableRight) {
    this->left = new Variable(variableLeft);
    this->op = op;
    this->right = new Variable(variableRight);
}

BinaryExpr::BinaryExpr(int literalIntLeft, BinaryOperator op, const std::string &variableRight) {
    this->left = new LiteralInt(literalIntLeft);
    this->op = op;
    this->right = new Variable(variableRight);
}

BinaryExpr::BinaryExpr(const std::string &variableLeft, BinaryOperator op, int literalIntRight) {
    this->left = new Variable(variableLeft);
    this->op = op;
    this->right = new LiteralInt(literalIntRight);
}

BinaryExpr::BinaryExpr(int literalIntLeft, BinaryOperator op, int literalIntRight) {
    this->left = new LiteralInt(literalIntLeft);
    this->op = op;
    this->right = new LiteralInt(literalIntRight);
}
