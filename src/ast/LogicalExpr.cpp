#include "../../include/ast/LogicalExpr.h"
#include <Variable.h>


json LogicalExpr::toJson() const {
    json j;
    j["type"] = getNodeName();
    j["leftOperand"] = this->left->toJson();
    j["operator"] = this->op->getOperatorString();
    j["rightOperand"] = this->right->toJson();
    return j;
}


LogicalExpr::LogicalExpr(AbstractExpr *left, OpSymb::LogCompOp op, AbstractExpr *right) :
        left(left), right(right) {
    this->op = new Operator(op);
}

void LogicalExpr::accept(Visitor &v) {
    v.visit(*this);
}

AbstractExpr *LogicalExpr::getLeft() const {
    return left;
}

Operator *LogicalExpr::getOp() const {
    return op;
}

AbstractExpr *LogicalExpr::getRight() const {
    return right;
}

std::string LogicalExpr::getNodeName() const {
    return "LogicalExpr";
}

