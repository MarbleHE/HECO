#include <vector>
#include "ast_opt/ast/UnaryExpression.h"
#include "ast_opt/visitor/IVisitor.h"

UnaryExpression::~UnaryExpression() = default;

UnaryExpression::UnaryExpression(std::unique_ptr<AbstractExpression> Operand, Operator op)
    : operand(std::move(operand)), op(op) {}

UnaryExpression::UnaryExpression(const UnaryExpression &other)
    : operand(other.operand ? other.operand->clone() : nullptr),      op(other.op) {}

UnaryExpression::UnaryExpression(UnaryExpression &&other) noexcept
    : operand(std::move(other.operand)),
      op(other.op) {}

UnaryExpression &UnaryExpression::operator=(const UnaryExpression &other) {
  operand = other.operand ? other.operand->clone() : nullptr;
  op = other.op;
  return *this;
}

UnaryExpression &UnaryExpression::operator=(UnaryExpression &&other) noexcept {
  operand = std::move(other.operand);
  op = other.op;
  return *this;
}

std::unique_ptr<UnaryExpression> UnaryExpression::clone() const {
  return std::unique_ptr<UnaryExpression>(clone_impl());
}

bool UnaryExpression::hasOperand() const {
  return operand!=nullptr;
}

bool UnaryExpression::hasOperator() const {
  return true;
}

AbstractExpression &UnaryExpression::getOperand() {
  if (hasOperand()) {
    return *operand;
  } else {
    throw std::runtime_error("Cannot get null operand.");
  }
}

const AbstractExpression &UnaryExpression::getOperand() const {
  if (hasOperand()) {
    return *operand;
  } else {
    throw std::runtime_error("Cannot get null operand.");
  }
}

Operator &UnaryExpression::getOperator() {
  return op;
}

const Operator &UnaryExpression::getOperator() const {
  return op;
}

void UnaryExpression::setOperand(std::unique_ptr<AbstractExpression> newOperand) {
  operand = std::move(newOperand);
}

void UnaryExpression::setOperator(Operator newOperator) {
  op = newOperator;
}


///////////////////////////////////////////////
////////// AbstractNode Interface /////////////
///////////////////////////////////////////////

UnaryExpression *UnaryExpression::clone_impl() const {
  return new UnaryExpression(*this);
}

void UnaryExpression::accept(IVisitor &v) {
  v.visit(*this);
}

AbstractNode::iterator UnaryExpression::begin() {
  return AbstractNode::iterator(std::make_unique<UnaryExpressionIteratorImpl<AbstractNode>>(*this, 0));
}

AbstractNode::const_iterator UnaryExpression::begin() const {
  return AbstractNode::const_iterator(std::make_unique<UnaryExpressionIteratorImpl<const AbstractNode>>(*this, 0));
}

AbstractNode::iterator UnaryExpression::end() {
  return AbstractNode::iterator(std::make_unique<UnaryExpressionIteratorImpl<AbstractNode>>(*this,
                                                                                            countChildren()));
}

AbstractNode::const_iterator UnaryExpression::end() const {
  return AbstractNode::const_iterator(std::make_unique<UnaryExpressionIteratorImpl<const AbstractNode>>(*this,
                                                                                                        countChildren()));
}

size_t UnaryExpression::countChildren() const {
  return hasOperand();
}

nlohmann::json UnaryExpression::toJson() const {
  nlohmann::json j = {{"type", getNodeType()}};
  if (hasOperand()) j["operand"] = getOperand().toJson();
  j["operator"] = op.toString();

  return j;
}

std::string UnaryExpression::toString(bool printChildren) const {
  return AbstractNode::toStringHelper(printChildren, {op.toString()});

}
std::string UnaryExpression::getNodeType() const {
  return "UnaryExpression";
}
