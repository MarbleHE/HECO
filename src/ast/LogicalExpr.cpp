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

void LogicalExpr::accept(Visitor &v) {
    v.visit(*this);
}

AbstractExpr *LogicalExpr::getLeft() const {
    return left;
}

Operator &LogicalExpr::getOp() const {
    return *op;
}

AbstractExpr *LogicalExpr::getRight() const {
    return right;
}

std::string LogicalExpr::getNodeName() const {
    return "LogicalExpr";
}

LogicalExpr::~LogicalExpr() {
    delete left;
    delete right;
    delete op;
}

Literal *LogicalExpr::evaluate(Ast &ast) {
    // we first need to evaluate the left-handside and right-handside as they can consists of nested binary expressions
    return this->getOp().applyOperator(this->getLeft()->evaluate(ast), this->getRight()->evaluate(ast));
}

LogicalExpr::LogicalExpr() : left(nullptr), op(nullptr), right(nullptr) {}

LogicalExpr::LogicalExpr(OpSymb::LogCompOp op) : left(nullptr), right(nullptr) {
    this->op = new Operator(op);
}

std::vector<std::string> LogicalExpr::getVariableIdentifiers() {
    auto leftVec = left->getVariableIdentifiers();
    auto rightVec = right->getVariableIdentifiers();
    leftVec.reserve(leftVec.size() + rightVec.size());
    leftVec.insert(leftVec.end(), rightVec.begin(), rightVec.end());
    return leftVec;
}

LogicalExpr *LogicalExpr::contains(LogicalExpr *lexpTemplate, AbstractExpr *excludedSubtree) {
    if (excludedSubtree != nullptr && this == excludedSubtree) {
        return nullptr;
    } else {
        bool emptyOrEqualLeft = (!lexpTemplate->getLeft() || lexpTemplate->getLeft() == this->getLeft());
        bool emptyOrEqualRight = (!lexpTemplate->getRight() || lexpTemplate->getRight() == this->getRight());
        bool emptyOrEqualOp = (lexpTemplate->getOp().isUndefined() || this->getOp() == lexpTemplate->getOp());
        return (emptyOrEqualLeft && emptyOrEqualRight && emptyOrEqualOp) ? this : nullptr;
    }
}

int LogicalExpr::countByTemplate(AbstractExpr *abstractExpr) {
    // check if abstractExpr is of type BinaryExpr
    if (auto expr = dynamic_cast<LogicalExpr *>(abstractExpr)) {
        // check if current BinaryExpr fulfills requirements of template abstractExpr
        // also check left and right operands for nested BinaryExps
        return (this->contains(expr, nullptr) != nullptr ? 1 : 0)
               + left->countByTemplate(abstractExpr)
               + right->countByTemplate(abstractExpr);
    } else {
        return 0;
    }
}

Node *LogicalExpr::clone() {
    auto clonedLexp = new LogicalExpr();
    clonedLexp->setUniqueNodeId(this->getUniqueNodeId());
    return clonedLexp;
}


void LogicalExpr::setOperands(AbstractExpr *lexp, AbstractExpr *rexp) {
    this->removeChildren();
    this->addChildren({lexp, op, rexp});
    this->left = lexp;
    this->right = rexp;
}
