#include "ast_opt/ast/Call.h"
#include "ast_opt/utilities/IVisitor.h"

/// Convenience typedef for conciseness
typedef std::unique_ptr<AbstractExpression> exprPtr;

Call::~Call() = default;

Call::Call(std::string identifier, std::vector<std::unique_ptr<AbstractExpression>> &&arguments)
    : identifier(std::move(identifier)), arguments(std::move(arguments)) {}

Call::Call(const Call &other) : identifier(other.identifier) {
  // deep-copy the arguments, including nullptrs
  arguments.reserve(other.arguments.size());
  for (auto &a: other.arguments) {
    arguments.emplace_back(a ? a->clone(this) : nullptr);
  }

}

Call::Call(Call &&other) noexcept: identifier(std::move(other.identifier)), arguments(std::move(other.arguments)) {}

Call &Call::operator=(const Call &other) {
  identifier = other.identifier;
  // deep-copy the statements, including nullptrs
  arguments.clear();
  arguments.reserve(other.arguments.size());
  for (auto &a: other.arguments) {
    arguments.emplace_back(a ? a->clone(this) : nullptr);
  }
  return *this;
}
Call &Call::operator=(Call &&other) noexcept {
  identifier = std::move(other.identifier);
  arguments = std::move(other.arguments);
  return *this;
}
std::unique_ptr<Call> Call::clone(AbstractNode* parent_) const {
  return std::unique_ptr<Call>(clone_impl(parent_));
}

std::string Call::getIdentifier() const {
  return identifier;
}

std::vector<std::reference_wrapper<const AbstractExpression>> Call::getArguments() const {
  std::vector<std::reference_wrapper<const AbstractExpression>> r;
  for (auto &a: arguments) {
    if (a) { r.emplace_back(*a); }
  }
  return r;
}
std::vector<std::reference_wrapper<AbstractExpression>> Call::getArguments() {
  std::vector<std::reference_wrapper<AbstractExpression>> r;
  for (auto &a: arguments) {
    if (a) { r.emplace_back(*a); }
  }
  return r;
}

///////////////////////////////////////////////
////////// AbstractNode Interface /////////////
///////////////////////////////////////////////
Call *Call::clone_impl(AbstractNode* parent_) const {
  auto p = new Call(*this);
  if(parent_) {p->setParent(*parent_);}
  return p;
}

void Call::accept(IVisitor &v) {
  v.visit(*this);
}

AbstractNode::iterator Call::begin() {
  return AbstractNode::iterator(std::make_unique<CallIteratorImpl<AbstractNode>>(*this,
                                                                                 arguments.begin(),
                                                                                 arguments.end()));
}

AbstractNode::const_iterator Call::begin() const {
  return AbstractNode::const_iterator(std::make_unique<CallIteratorImpl<const AbstractNode>>(*this,
                                                                                             arguments.begin(),
                                                                                             arguments.end()));
}

AbstractNode::iterator Call::end() {
  return AbstractNode::iterator(std::make_unique<CallIteratorImpl<AbstractNode>>(*this,
                                                                                 arguments.end(),
                                                                                 arguments.end()));
}

AbstractNode::const_iterator Call::end() const {
  return AbstractNode::const_iterator(std::make_unique<CallIteratorImpl<const AbstractNode>>(*this,
                                                                                             arguments.end(),
                                                                                             arguments.end()));
}

size_t Call::countChildren() const {
  // Only non-null entries in the vector are counted as children
  // Because std::unique_ptr doesn't have copy, we can't use std::count_if
  size_t count = 0;
  for (auto &a : arguments) {
    if (a!=nullptr) { count++; }
  }
  return count;
}

nlohmann::json Call::toJson() const {
  std::vector<std::reference_wrapper<const AbstractExpression>> args = getArguments();
  std::vector<nlohmann::json> argsJSON;
  for (const AbstractExpression &a: args) {
    argsJSON.push_back(a.toJson());
  }
  nlohmann::json j = {{"type", getNodeType()},
                      {"identifier", getIdentifier()},
                      {"arguments", argsJSON}
  };
  return j;
}

std::string Call::toString(bool printChildren) const {
  return AbstractNode::toStringHelper(printChildren, {getIdentifier()});
}

std::string Call::getNodeType() const {
  return "Call";
}

std::unique_ptr<AbstractNode> Call::replaceChild(const AbstractNode &child, std::unique_ptr<AbstractNode> &&new_child) {
  for (auto &a: arguments) {
    if (a && &child==&*a) {
      auto t = std::move(a);
      if (!dynamic_pointer_move(a, new_child)) {
        a = std::move(t);
        throw std::runtime_error("Cannot replace child: Types not compatible");
      }
      t->setParent(nullptr);
      a->setParent(this);
      return t;
    }
  }

  throw std::runtime_error("Cannot replace child: not found!");
}
