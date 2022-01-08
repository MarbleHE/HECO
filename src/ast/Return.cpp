#include <utility>
#include "ast_opt/ast_parser/Parser.h"
#include "ast_opt/ast/Return.h"
#include "ast_opt/ast_utilities/IVisitor.h"

Return::~Return() = default;

Return::Return(std::unique_ptr<AbstractExpression> value) : value(std::move(value)) {}

Return::Return(const Return &other) : value(other.value ? other.value->clone(this) : nullptr) {}

Return::Return(Return &&other) noexcept: value(std::move(other.value)) {}

Return &Return::operator=(const Return &other) {
  value = other.value ? other.value->clone(this) : nullptr;
  return *this;
}

Return &Return::operator=(Return &&other) noexcept {
  value = std::move(other.value);
  return *this;
}

std::unique_ptr<Return> Return::clone(AbstractNode *parent_) const {
  return std::unique_ptr<Return>(clone_impl(parent_));
}

bool Return::hasValue() const {
  return value!=nullptr;
}
AbstractExpression &Return::getValue() {
  if (hasValue()) {
    return *value;
  } else {
    throw std::runtime_error("Cannot get null value.");
  }
}

const AbstractExpression &Return::getValue() const {
  if (hasValue()) {
    return *value;
  } else {
    throw std::runtime_error("Cannot get null value.");
  }
}

void Return::setValue(std::unique_ptr<AbstractExpression> newValue) {
  value = std::move(newValue);
}

///////////////////////////////////////////////
////////// AbstractNode Interface /////////////
///////////////////////////////////////////////
Return *Return::clone_impl(AbstractNode *parent_) const {
  auto p = new Return(*this);
  if (parent_) { p->setParent(*parent_); }
  return p;
}

void Return::accept(IVisitor &v) {
  v.visit(*this);
}
AbstractNode::iterator Return::begin() {
  return AbstractNode::iterator(std::make_unique<ReturnIteratorImpl<AbstractNode>>(*this, 0));
}

AbstractNode::const_iterator Return::begin() const {
  return AbstractNode::const_iterator(std::make_unique<ReturnIteratorImpl<const AbstractNode>>(*this, 0));
}

AbstractNode::iterator Return::end() {
  return AbstractNode::iterator(std::make_unique<ReturnIteratorImpl<AbstractNode>>(*this, countChildren()));
}

AbstractNode::const_iterator Return::end() const {
  return AbstractNode::const_iterator(std::make_unique<ReturnIteratorImpl<const AbstractNode>>(*this, countChildren()));
}

size_t Return::countChildren() const {
  return hasValue();
}

nlohmann::json Return::toJson() const {
  nlohmann::json j = {{"type", getNodeType()}};
  if (hasValue()) {
    j["value"] = getValue().toJson();
  }
  return j;
}

std::unique_ptr<Return> Return::fromJson(nlohmann::json j) {
  auto expression = Parser::parseJsonExpression(j["value"]);
  return std::make_unique<Return>(std::move(expression));
}

std::string Return::toString(bool printChildren) const {
  return AbstractNode::toStringHelper(printChildren, {});
}

std::string Return::getNodeType() const {
  return "Return";
}