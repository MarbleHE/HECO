#include "ast_opt/ast/TernaryOperator.h"
#include "ast_opt/utilities/IVisitor.h"

TernaryOperator::~TernaryOperator() = default;

TernaryOperator::TernaryOperator(std::unique_ptr<AbstractExpression> &&condition,
                                 std::unique_ptr<AbstractExpression> &&thenExpr,
                                 std::unique_ptr<AbstractExpression> &&elseExpr)
    : condition(std::move(condition)), thenExpr(std::move(thenExpr)), elseExpr(std::move(elseExpr)) {}

TernaryOperator::TernaryOperator(const TernaryOperator &other) : condition(other.condition ? other.condition->clone(this) : nullptr),
                                                                 thenExpr(other.thenExpr ? other.thenExpr->clone(this) : nullptr),
                                                                 elseExpr(other.elseExpr ? other.elseExpr->clone(this) : nullptr) {}

TernaryOperator::TernaryOperator(TernaryOperator &&other) noexcept: condition(std::move(other.condition)),
                                                                    thenExpr(std::move(other.thenExpr)),
                                                                    elseExpr(std::move(other.elseExpr)) {}

TernaryOperator &TernaryOperator::operator=(const TernaryOperator &other) {
  condition = other.condition ? other.condition->clone(this) : nullptr;
  thenExpr = other.thenExpr ? other.thenExpr->clone(this) : nullptr;
  elseExpr = other.elseExpr ? other.elseExpr->clone(this) : nullptr;
  return *this;
}

TernaryOperator &TernaryOperator::operator=(TernaryOperator &&other) noexcept {
  condition = std::move(other.condition);
  thenExpr = std::move(other.thenExpr);
  elseExpr = std::move(other.elseExpr);
  return *this;
}

std::unique_ptr<TernaryOperator> TernaryOperator::clone(AbstractNode *parent_) const {
  return std::unique_ptr<TernaryOperator>(clone_impl(parent_));
}

bool TernaryOperator::hasCondition() const {
  return condition!=nullptr;
}

bool TernaryOperator::hasThenExpr() const {
  return thenExpr!=nullptr;
}

bool TernaryOperator::hasElseExpr() const {
  return elseExpr!=nullptr;
}

AbstractExpression &TernaryOperator::getCondition() {
  if (hasCondition()) {
    return *condition;
  } else {
    throw std::runtime_error("Cannot get null condition.");
  }
}

const AbstractExpression &TernaryOperator::getCondition() const {
  if (hasCondition()) {
    return *condition;
  } else {
    throw std::runtime_error("Cannot get null condition.");
  }
}

AbstractExpression & TernaryOperator::getThenExpr() {
  if (hasThenExpr()) {
    return *thenExpr;
  } else {
    throw std::runtime_error("Cannot get null thenExpr.");
  }
}

const AbstractExpression &TernaryOperator::getThenExpr() const {
  if (hasThenExpr()) {
    return *thenExpr;
  } else {
    throw std::runtime_error("Cannot get null thenExpr.");
  }
}

AbstractExpression &TernaryOperator::getElseExpr() {
  if (hasElseExpr()) {
    return *elseExpr;
  } else {
    throw std::runtime_error("Cannot get null elseExpr.");
  }
}

const AbstractExpression &TernaryOperator::getElseExpr() const {
  if (hasElseExpr()) {
    return *elseExpr;
  } else {
    throw std::runtime_error("Cannot get null elseExpr.");
  }
}

void TernaryOperator::setCondition(std::unique_ptr<AbstractExpression> &&newCondition) {
  condition = std::move(newCondition);
}

void TernaryOperator::setThenExpr(std::unique_ptr<AbstractExpression> &&newthenExpr) {
  thenExpr = std::move(newthenExpr);
}

void TernaryOperator::setElseExpr(std::unique_ptr<AbstractExpression> &&newelseExpr) {
  elseExpr = std::move(newelseExpr);
}

///////////////////////////////////////////////
////////// AbstractNode Interface /////////////
///////////////////////////////////////////////
TernaryOperator *TernaryOperator::clone_impl(AbstractNode *parent_) const {
  auto p = new TernaryOperator(*this);
  if (parent_) { p->setParent(*parent_); }
  return p;
}

void TernaryOperator::accept(IVisitor &v) {
  v.visit(*this);
}

AbstractNode::iterator TernaryOperator::begin() {
  return AbstractNode::iterator(std::make_unique<TernaryExpressionIteratorImpl<AbstractNode>>(*this, 0));
}

AbstractNode::const_iterator TernaryOperator::begin() const {
  return AbstractNode::const_iterator(std::make_unique<TernaryExpressionIteratorImpl<const AbstractNode>>(*this, 0));
}

AbstractNode::iterator TernaryOperator::end() {
  return AbstractNode::iterator(std::make_unique<TernaryExpressionIteratorImpl<AbstractNode>>(*this, countChildren()));
}

AbstractNode::const_iterator TernaryOperator::end() const {
  return AbstractNode::const_iterator(std::make_unique<TernaryExpressionIteratorImpl<const AbstractNode>>(*this, countChildren()));
}

size_t TernaryOperator::countChildren() const {
  return hasCondition() + hasThenExpr() + hasElseExpr();
}

nlohmann::json TernaryOperator::toJson() const {
  nlohmann::json j;
  j["type"] = getNodeType();
  if (hasCondition()) j["condition"] = getCondition().toJson();
  if (hasThenExpr()) j["thenExpr"] = getThenExpr().toJson();
  if (hasElseExpr()) j["elseExpr"] = getElseExpr().toJson();
  return j;
}

std::string TernaryOperator::toString(bool printChildren) const {
  return AbstractNode::toStringHelper(printChildren, {});
}

std::string TernaryOperator::getNodeType() const {
  return "TernaryExpression";
}
std::unique_ptr<AbstractNode> TernaryOperator::replaceChild(const AbstractNode &child,
                                                            std::unique_ptr<AbstractNode> &&new_child) {
  if (condition && &child==&*condition) {
    auto t = std::move(condition);
    if (!dynamic_pointer_move(condition, new_child)) {
      condition = std::move(t);
      throw std::runtime_error("Cannot replace child: Types not compatible");
    }
    t->setParent(nullptr);
    condition->setParent(this);
    return t;
  } else if (thenExpr && &child==&*thenExpr) {
    auto t = std::move(thenExpr);
    if (!dynamic_pointer_move(thenExpr, new_child)) {
      thenExpr = std::move(t);
      throw std::runtime_error("Cannot replace child: Types not compatible");
    }
    t->setParent(nullptr);
    thenExpr->setParent(this);
    return t;
  } else if (elseExpr && &child==&*elseExpr) {
    auto t = std::move(elseExpr);
    if (!dynamic_pointer_move(elseExpr, new_child)) {
      elseExpr = std::move(t);
      throw std::runtime_error("Cannot replace child: Types not compatible");
    }
    t->setParent(nullptr);
    elseExpr->setParent(this);
    return t;
  } else {
    throw std::runtime_error("Cannot replace child: not found!");
  }
}
