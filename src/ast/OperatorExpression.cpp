#include <iostream>
#include <exception>
#include "ast_opt/ast/OperatorExpression.h"
#include "ast_opt/visitor/IVisitor.h"

/// Convenience typedef for conciseness
typedef std::unique_ptr<AbstractExpression> exprPtr;

OperatorExpression::~OperatorExpression() = default;

OperatorExpression::OperatorExpression(Operator op, std::vector<std::unique_ptr<AbstractExpression>> && operands) : op(std::move(op)), operands(std::move(operands)) {}

OperatorExpression::OperatorExpression(const OperatorExpression &other) : op(other.op) {
  // deep-copy the operands, including nullptrs
  operands.reserve(other.operands.size());
  for (auto &s: other.operands) {
    operands.emplace_back(s ? s->clone() : nullptr);
  }
}

OperatorExpression::OperatorExpression(OperatorExpression &&other) noexcept: op(std::move(other.op)), operands(std::move(other.operands)) {}

OperatorExpression &OperatorExpression::operator=(const OperatorExpression &other) {
  op = other.op;
  operands.clear();
  // deep-copy the operands, including nullptrs
  operands.reserve(other.operands.size());
  for (auto &s: other.operands) {
    operands.emplace_back(s ? s->clone() : nullptr);
  }
  return *this;
}
OperatorExpression &OperatorExpression::operator=(OperatorExpression &&other) noexcept {
  op = std::move(other.op);
  operands = std::move(other.operands);
  return *this;
}
std::unique_ptr<OperatorExpression> OperatorExpression::clone() const {
  return std::unique_ptr<OperatorExpression>(clone_impl());
}

bool OperatorExpression::hasNullOperands() {
  // Because std::unique_ptr doesn't have copy, we can't use std::count_if
  size_t count = 0;
  for (auto &s : operands) {
    if (s==nullptr) { count++; }
  }
  return count!=0;
}

std::vector<std::reference_wrapper<AbstractExpression>> OperatorExpression::getOperands() {
  std::vector<std::reference_wrapper<AbstractExpression>> r;
  for (auto &o: operands) {
    if (o!=nullptr) { r.emplace_back(*o); }
  }
  return r;
}

std::vector<std::reference_wrapper<const AbstractExpression>> OperatorExpression::getOperands() const {
  std::vector<std::reference_wrapper<const AbstractExpression>> r;
  for (auto &o: operands) {
    if (o!=nullptr) { r.emplace_back(*o); }
  }
  return r;
}

void OperatorExpression::appendOperand(std::unique_ptr<AbstractExpression> operand) {
  operands.emplace_back(std::move(operand));
}

void OperatorExpression::prependOperand(std::unique_ptr<AbstractExpression> operand) {
  operands.insert(operands.begin(),std::move(operand));
}

void OperatorExpression::removeNullOperands() {
  std::vector<exprPtr> new_operands;
  for (auto &o: operands) {
    if (o!=nullptr) { new_operands.emplace_back(std::move(o)); }
  }
  operands = std::move(new_operands);
}

///////////////////////////////////////////////
////////// AbstractNode Interface /////////////
///////////////////////////////////////////////
OperatorExpression *OperatorExpression::clone_impl() const {
  return new OperatorExpression(*this);
}

void OperatorExpression::accept(IVisitor &v) {
  v.visit(*this);
}

AbstractNode::iterator OperatorExpression::begin() {
  return AbstractNode::iterator(std::make_unique<OperatorExpressionIteratorImpl<AbstractNode>>(*this,
                                                                                  operands.begin(),
                                                                                  operands.end()));
}

AbstractNode::const_iterator OperatorExpression::begin() const {
  return AbstractNode::const_iterator(std::make_unique<OperatorExpressionIteratorImpl<const AbstractNode>>(*this,
                                                                                              operands.begin(),
                                                                                              operands.end()));
}

AbstractNode::iterator OperatorExpression::end() {
  return AbstractNode::iterator(std::make_unique<OperatorExpressionIteratorImpl<AbstractNode>>(*this,
                                                                                  operands.end(),
                                                                                  operands.end()));
}

AbstractNode::const_iterator OperatorExpression::end() const {
  return AbstractNode::const_iterator(std::make_unique<OperatorExpressionIteratorImpl<const AbstractNode>>(*this,
                                                                                              operands.end(),
                                                                                              operands.end()));
}

size_t OperatorExpression::countChildren() const {
  // Only non-null entries in the vector are counted as children
  // Because std::unique_ptr doesn't have copy, we can't use std::count_if
  size_t count = 0;
  for (auto &s : operands) {
    if (s!=nullptr) { count++; }
  }
  return count;
}

nlohmann::json OperatorExpression::toJson() const {
  std::vector<std::reference_wrapper<const AbstractExpression>> stmts = getOperands();
  std::vector<nlohmann::json> operandsJson;
  for(const AbstractExpression& o: stmts) {
    operandsJson.push_back(o.toJson());
 }
  nlohmann::json j = {{"type", getNodeType()},
                      {"operands", operandsJson}};
  return j;
}

std::string OperatorExpression::toString(bool printChildren) const {
  return AbstractNode::toStringHelper(printChildren, {op.toString()});
}

std::string OperatorExpression::getNodeType() const {
  return "OperatorExpression";
}



