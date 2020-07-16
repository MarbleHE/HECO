#include "ast_opt/ast/VariableAssignment.h"
#include "ast_opt/ast/Variable.h"
#include "ast_opt/visitor/IVisitor.h"

VariableAssignment::~VariableAssignment() = default;

VariableAssignment::VariableAssignment(std::unique_ptr<AbstractTarget> target,
                                       std::unique_ptr<AbstractExpression> value)
    : target(std::move(target)), value(std::move(value)) {}

VariableAssignment::VariableAssignment(const VariableAssignment &other)
    : target(other.target ? other.target->clone() : nullptr), value(other.value ? other.value->clone() : nullptr) {};

VariableAssignment::VariableAssignment(VariableAssignment &&other)
noexcept: target(std::move(other.target)), value(std::move(other.value)) {};

VariableAssignment &VariableAssignment::operator=(const VariableAssignment &other) {
  target = other.target ? other.target->clone() : nullptr;
  value = other.value ? other.value->clone() : nullptr;
  return *this;
}

VariableAssignment &VariableAssignment::operator=(VariableAssignment &&other) noexcept {
  target = std::move(other.target);
  value = std::move(other.value);
  return *this;
}

std::unique_ptr<VariableAssignment> VariableAssignment::clone() const {
  return std::unique_ptr<VariableAssignment>(clone_impl());
}

bool VariableAssignment::hasTarget() const {
  return target!=nullptr;
}

bool VariableAssignment::hasValue() const {
  return value!=nullptr;
}

AbstractTarget &VariableAssignment::getTarget() {
  if (hasTarget()) {
    return *target;
  } else {
    throw std::runtime_error("Cannot get null target.");
  }
}

const AbstractTarget &VariableAssignment::getTarget() const {
  if (hasTarget()) {
    return *target;
  } else {
    throw std::runtime_error("Cannot get null target.");
  }
}

AbstractExpression &VariableAssignment::getValue() {
  if (hasValue()) {
    return *value;
  } else {
    throw std::runtime_error("Cannot get null value.");
  }
}

const AbstractExpression &VariableAssignment::getValue() const {
  if (hasValue()) {
    return *value;
  } else {
    throw std::runtime_error("Cannot get null value.");
  }
}

void VariableAssignment::setTarget(std::unique_ptr<AbstractTarget> newTarget) {
  target = std::move(newTarget);
}

void VariableAssignment::setValue(std::unique_ptr<AbstractExpression> newValue) {
  value = std::move(newValue);
}

///////////////////////////////////////////////
////////// AbstractNode Interface /////////////
///////////////////////////////////////////////
VariableAssignment *VariableAssignment::clone_impl() const {
  return new VariableAssignment(*this);
}

void VariableAssignment::accept(IVisitor &v) {
  v.visit(*this);
}

AbstractNode::iterator VariableAssignment::begin() {
  return AbstractNode::iterator(std::make_unique<VariableAssignmentIteratorImpl<AbstractNode>>(*this, 0));
}

AbstractNode::const_iterator VariableAssignment::begin() const {
  return AbstractNode::const_iterator(std::make_unique<VariableAssignmentIteratorImpl<const AbstractNode>>(*this, 0));
}

AbstractNode::iterator VariableAssignment::end() {
  return AbstractNode::iterator(std::make_unique<VariableAssignmentIteratorImpl<AbstractNode>>(*this, countChildren()));
}

AbstractNode::const_iterator VariableAssignment::end() const {
  return AbstractNode::const_iterator(std::make_unique<VariableAssignmentIteratorImpl<const AbstractNode>>(*this,
                                                                                                           countChildren()));
}

size_t VariableAssignment::countChildren() const {
  return hasValue() + hasTarget();
}

nlohmann::json VariableAssignment::toJson() const {
  nlohmann::json j;
  j["type"] = getNodeType();
  j["target"] = target ? target->toJson() : "";
  j["value"] = value ? value->toJson() : "";
  return j;
}

std::string VariableAssignment::toString(bool printChildren) const {
  return AbstractNode::toStringHelper(printChildren, {});
}

std::string VariableAssignment::getNodeType() const {
  return "VariableAssignment";
}
