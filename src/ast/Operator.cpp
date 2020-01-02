#include <variant>
#include "../../include/ast/Operator.h"

void Operator::accept(Visitor &v) {
    v.visit(*this);
}

Operator::Operator(OpSymb::LogCompOp op) : operatorString(OpSymb::getTextRepr(op)) {}

Operator::Operator(OpSymb::BinaryOp op) : operatorString(OpSymb::getTextRepr(op)) {}

Operator::Operator(OpSymb::UnaryOp op) : operatorString(OpSymb::getTextRepr(op)) {}

const std::string &Operator::getOperatorString() const {
    return operatorString;
}

std::string Operator::getNodeName() const {
    return "Operator";
}

bool Operator::isUndefined() {
    return operatorString.empty();
}

bool Operator::operator==(const Operator &rhs) const {
    return operatorString == rhs.operatorString;
}

bool Operator::operator!=(const Operator &rhs) const {
    return !(rhs == *this);
}

bool Operator::equals(std::variant<OpSymb::BinaryOp, OpSymb::LogCompOp, OpSymb::UnaryOp> op) const {
    return this->getOperatorString() == OpSymb::getTextRepr(op);
}
