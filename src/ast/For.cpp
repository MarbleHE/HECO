#include "ast_opt/ast_parser/Parser.h"
#include "ast_opt/ast_utilities/NodeUtils.h"
#include "ast_opt/ast/For.h"
#include "ast_opt/ast_utilities/IVisitor.h"

For::~For() {

}
For::For(std::unique_ptr<Block> initializer,
         std::unique_ptr<AbstractExpression> condition,
         std::unique_ptr<Block> update,
         std::unique_ptr<Block> body)
    : initializer(std::move(initializer)),
      condition(std::move(condition)),
      update(std::move(update)),
      body(std::move(body)) {}

For::For(const For &other)
    : initializer(other.initializer ? other.initializer->clone(this) : nullptr),
      condition(other.condition ? other.condition->clone(this) : nullptr),
      update(other.update ? other.update->clone(this) : nullptr),
      body(other.body ? other.body->clone(this) : nullptr) {}

For::For(For &&other) noexcept
    : initializer(std::move(other.initializer)),
      condition(std::move(other.condition)),
      update(std::move(other.update)),
      body(std::move(other.body)) {}

For &For::operator=(const For &other) {
  initializer = other.initializer ? other.initializer->clone(this) : nullptr;
  condition = other.condition ? other.condition->clone(this) : nullptr;
  update = other.update ? other.update->clone(this) : nullptr;
  body = other.body ? other.body->clone(this) : nullptr;
  return *this;
}

For &For::operator=(For &&other) noexcept {
  initializer = std::move(other.initializer);
  condition = std::move(other.condition);
  update = std::move(other.update);
  body = std::move(other.body);
  return *this;
}

std::unique_ptr<For> For::clone(AbstractNode* parent_) const {
  return std::unique_ptr<For>(clone_impl(parent_));
}

bool For::hasInitializer() const {
  return initializer!=nullptr;
}

bool For::hasCondition() const {
  return condition!=nullptr;
}

bool For::hasUpdate() const {
  return update!=nullptr;
}

bool For::hasBody() const {
  return body!=nullptr;
}

Block &For::getInitializer() {
  if (hasInitializer()) {
    return *initializer;
  } else {
    throw std::runtime_error("Cannot get null initializer.");
  }
}

const Block &For::getInitializer() const {
  if (hasInitializer()) {
    return *initializer;
  } else {
    throw std::runtime_error("Cannot get null initializer.");
  }
}

AbstractExpression &For::getCondition() {
  if (hasCondition()) {
    return *condition;
  } else {
    throw std::runtime_error("Cannot get null condition.");
  }
}

const AbstractExpression &For::getCondition() const {
  if (hasCondition()) {
    return *condition;
  } else {
    throw std::runtime_error("Cannot get null condition.");
  }
}

Block &For::getUpdate() {
  if (hasUpdate()) {
    return *update;
  } else {
    throw std::runtime_error("Cannot get null update.");
  }
}

const Block &For::getUpdate() const {
  if (hasUpdate()) {
    return *update;
  } else {
    throw std::runtime_error("Cannot get null update.");
  }
}

Block &For::getBody() {
  if (hasBody()) {
    return *body;
  } else {
    throw std::runtime_error("Cannot get null body.");
  }
}

const Block &For::getBody() const {
  if (hasBody()) {
    return *body;
  } else {
    throw std::runtime_error("Cannot get null body.");
  }
}

void For::setInitializer(std::unique_ptr<Block> newInitializer) {
  initializer = std::move(newInitializer);
}

void For::setCondition(std::unique_ptr<AbstractExpression> newCondition) {
  condition = std::move(newCondition);
}

void For::setUpdate(std::unique_ptr<Block> newUpdate) {
  update = std::move(newUpdate);
}

void For::setBody(std::unique_ptr<Block> newBody) {
  body = std::move(newBody);
}

///////////////////////////////////////////////
////////// AbstractNode Interface /////////////
///////////////////////////////////////////////
For *For::clone_impl(AbstractNode* parent_) const {
  auto p = new For(*this);
  if(parent_) {p->setParent(*parent_);}
  return p;
}
void For::accept(IVisitor &v) {
  v.visit(*this);
}

AbstractNode::iterator For::begin() {
  return AbstractNode::iterator(std::make_unique<ForIteratorImpl<AbstractNode>>
                                    (*this, 0));
}

AbstractNode::const_iterator For::begin() const {
  return AbstractNode::const_iterator(std::make_unique<ForIteratorImpl<const AbstractNode>>(*this, 0));
}

AbstractNode::iterator For::end() {
  return AbstractNode::iterator(std::make_unique<ForIteratorImpl<AbstractNode>>
                                    (*this, countChildren()));
}

AbstractNode::const_iterator For::end() const {
  return AbstractNode::const_iterator(std::make_unique<ForIteratorImpl<const AbstractNode>>(*this, countChildren()));
}
size_t For::countChildren() const {
  return size_t(hasInitializer()) + hasCondition() + hasUpdate() + hasBody();
}

nlohmann::json For::toJson() const {
  nlohmann::json j;
  j["type"] = getNodeType();
  if (hasInitializer()) j["initializer"] = getInitializer().toJson();
  if (hasCondition()) j["condition"] = getCondition().toJson();
  if (hasUpdate()) j["update"] = getUpdate().toJson();
  if (hasBody()) j["body"] = getBody().toJson();
  return j;
}

std::unique_ptr<For> For::fromJson(nlohmann::json j) {
  auto initializerStmt = (j.contains("initializer")) ? Parser::parseJsonStatement(j["initializer"]) : nullptr;
  auto conditionExpr = (j.contains("condition")) ? Parser::parseJsonExpression(j["condition"]) : nullptr;
  auto updateStmt = (j.contains("update")) ? Parser::parseJsonStatement(j["update"]) : nullptr;
  auto bodyStmt = (j.contains("body")) ? Parser::parseJsonStatement(j["body"]) : nullptr;

  auto initializerBlock = castUniquePtr<AbstractStatement, Block>(std::move(initializerStmt));
  auto updateBlock = castUniquePtr<AbstractStatement, Block>(std::move(updateStmt));
  auto bodyBlock = castUniquePtr<AbstractStatement, Block>(std::move(bodyStmt));

  return std::make_unique<For>(std::move(initializerBlock), std::move(conditionExpr), std::move(updateBlock), std::move(bodyBlock));
}

std::string For::toString(bool printChildren) const {
  return AbstractNode::toStringHelper(printChildren, {});
}

std::string For::getNodeType() const {
  return "For";
}