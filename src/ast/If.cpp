#include "ast_opt/ast/If.h"
#include "ast_opt/utilities/IVisitor.h"

If::~If() = default;

If::If(std::unique_ptr<AbstractExpression> &&condition,
       std::unique_ptr<Block> &&thenBranch,
       std::unique_ptr<Block> &&elseBranch)
    : condition(std::move(condition)), thenBranch(std::move(thenBranch)), elseBranch(std::move(elseBranch)) {}

If::If(const If &other) : condition(other.condition ? other.condition->clone(this) : nullptr),
                          thenBranch(other.thenBranch ? other.thenBranch->clone(this) : nullptr),
                          elseBranch(other.elseBranch ? other.elseBranch->clone(this) : nullptr) {}

If::If(If &&other) noexcept: condition(std::move(other.condition)),
                             thenBranch(std::move(other.thenBranch)),
                             elseBranch(std::move(other.elseBranch)) {}

If &If::operator=(const If &other) {
  condition = other.condition ? other.condition->clone(this) : nullptr;
  thenBranch = other.thenBranch ? other.thenBranch->clone(this) : nullptr;
  elseBranch = other.elseBranch ? other.elseBranch->clone(this) : nullptr;
  return *this;
}

If &If::operator=(If &&other) noexcept {
  condition = std::move(other.condition);
  thenBranch = std::move(other.thenBranch);
  elseBranch = std::move(other.elseBranch);
  return *this;
}
std::unique_ptr<If> If::clone(AbstractNode *parent_) const {
  return std::unique_ptr<If>(clone_impl(parent_));
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
If *If::clone_impl(AbstractNode *parent_) const {
  auto p = new If(*this);
  if (parent_) { p->setParent(*parent_); }
  return p;
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
std::unique_ptr<AbstractNode> If::replaceChild(const AbstractNode &child, std::unique_ptr<AbstractNode> &&new_child) {
  if (condition && &child==&*condition) {
    auto t = std::move(condition);
    if (!dynamic_pointer_move(condition, new_child)) {
      throw std::runtime_error("Cannot replace child: Types not compatible");
    }
    return t;
  } else if (thenBranch && &child==&*thenBranch) {
    auto t = std::move(thenBranch);
    if (!dynamic_pointer_move(thenBranch, new_child)) {
      throw std::runtime_error("Cannot replace child: Types not compatible");
    }
    return t;
  } else if (elseBranch && &child==&*elseBranch) {
    auto t = std::move(elseBranch);
    if (!dynamic_pointer_move(elseBranch, new_child)) {
      throw std::runtime_error("Cannot replace child: Types not compatible");
    }
    return t;
  } else {
    throw std::runtime_error("Cannot replace child: not found!");
  }
}
