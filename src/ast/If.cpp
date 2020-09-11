#include "ast_opt/ast/If.h"
#include "ast_opt/visitor/IVisitor.h"

If::~If() = default;

If::If(std::unique_ptr<AbstractExpression> &&condition,
       std::unique_ptr<Block> &&thenBranch,
       std::unique_ptr<Block> &&elseBranch)
    : condition(std::move(condition)), thenBranch(std::move(thenBranch)), elseBranch(std::move(elseBranch)) {}

If::If(const If &other) : condition(other.condition ? other.condition->clone() : nullptr),
                          thenBranch(other.thenBranch ? other.thenBranch->clone() : nullptr),
                          elseBranch(other.elseBranch ? other.elseBranch->clone() : nullptr) {}

If::If(If &&other) noexcept: condition(std::move(other.condition)),
                             thenBranch(std::move(other.thenBranch)),
                             elseBranch(std::move(other.elseBranch)) {}

If &If::operator=(const If &other) {
  condition = other.condition ? other.condition->clone() : nullptr;
  thenBranch = other.thenBranch ? other.thenBranch->clone() : nullptr;
  elseBranch = other.elseBranch ? other.elseBranch->clone() : nullptr;
  return *this;
}

If &If::operator=(If &&other) noexcept {
  condition = std::move(other.condition);
  thenBranch = std::move(other.thenBranch);
  elseBranch = std::move(other.elseBranch);
  return *this;
}
std::unique_ptr<If> If::clone() const {
  return std::unique_ptr<If>(clone_impl());
}

bool If::hasCondition() const {
  return condition!=nullptr;
}

bool If::hasThenBranch() const {
  return thenBranch!=nullptr;
}

bool If::hasElseBranch() const {
  return elseBranch!=nullptr;
}

AbstractExpression &If::getCondition() {
  if (hasCondition()) {
    return *condition;
  } else {
    throw std::runtime_error("Cannot get null condition.");
  }
}

const AbstractExpression &If::getCondition() const {
  if (hasCondition()) {
    return *condition;
  } else {
    throw std::runtime_error("Cannot get null condition.");
  }
}

Block &If::getThenBranch() {
  if (hasThenBranch()) {
    return *thenBranch;
  } else {
    throw std::runtime_error("Cannot get null thenBranch.");
  }
}

const Block &If::getThenBranch() const {
  if (hasThenBranch()) {
    return *thenBranch;
  } else {
    throw std::runtime_error("Cannot get null thenBranch.");
  }
}

Block &If::getElseBranch() {
  if (hasElseBranch()) {
    return *elseBranch;
  } else {
    throw std::runtime_error("Cannot get null elseBranch.");
  }
}

const Block &If::getElseBranch() const {
  if (hasElseBranch()) {
    return *elseBranch;
  } else {
    throw std::runtime_error("Cannot get null elseBranch.");
  }
}

void If::setCondition(std::unique_ptr<AbstractExpression> &&newCondition) {
  condition = std::move(newCondition);
}

void If::setThenBranch(std::unique_ptr<Block> &&newThenBranch) {
  thenBranch = std::move(newThenBranch);
}

void If::setElseBranch(std::unique_ptr<Block> &&newElseBranch) {
  elseBranch = std::move(newElseBranch);
}

///////////////////////////////////////////////
////////// AbstractNode Interface /////////////
///////////////////////////////////////////////
If *If::clone_impl() const {
  return new If(*this);
}

void If::accept(IVisitor &v) {
  v.visit(*this);
}

AbstractNode::iterator If::begin() {
  return AbstractNode::iterator(std::make_unique<IfIteratorImpl<AbstractNode>>(*this, 0));
}

AbstractNode::const_iterator If::begin() const {
  return AbstractNode::const_iterator(std::make_unique<IfIteratorImpl<const AbstractNode>>(*this, 0));
}

AbstractNode::iterator If::end() {
  return AbstractNode::iterator(std::make_unique<IfIteratorImpl<AbstractNode>>(*this, countChildren()));
}

AbstractNode::const_iterator If::end() const {
  return AbstractNode::const_iterator(std::make_unique<IfIteratorImpl<const AbstractNode>>(*this, countChildren()));
}

size_t If::countChildren() const {
  return hasCondition() + hasThenBranch() + hasElseBranch();
}

nlohmann::json If::toJson() const {
  nlohmann::json j;
  j["type"] = getNodeType();
  if (hasCondition()) j["condition"] = getCondition().toJson();
  if (hasThenBranch()) j["thenBranch"] = getThenBranch().toJson();
  if (hasElseBranch()) j["elseBranch"] = getElseBranch().toJson();
  return j;
}

std::string If::toString(bool printChildren) const {
  return AbstractNode::toStringHelper(printChildren, {});
}

std::string If::getNodeType() const {
  return "If";
}