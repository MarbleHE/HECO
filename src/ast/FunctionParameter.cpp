#include <utility>
#include "ast_opt/ast/FunctionParameter.h"
#include "ast_opt/utilities/IVisitor.h"

FunctionParameter::~FunctionParameter() = default;

FunctionParameter::FunctionParameter(Datatype parameter_type,
                                     std::string identifier)
    : identifier(std::move(identifier)), parameter_type(std::move(parameter_type)) {}

FunctionParameter::FunctionParameter(const FunctionParameter &other) : identifier(other.identifier),
                                                                       parameter_type(other.parameter_type) {}

FunctionParameter::FunctionParameter(FunctionParameter &&other) noexcept: identifier(std::move(other.identifier)),
                                                                          parameter_type(std::move(other
                                                                                                       .parameter_type)) {}

FunctionParameter &FunctionParameter::operator=(const FunctionParameter &other) {
  identifier = other.identifier;
  return *this;
}
FunctionParameter &FunctionParameter::operator=(FunctionParameter &&other) noexcept {
  identifier = std::move(other.identifier);
  return *this;
}

std::unique_ptr<FunctionParameter> FunctionParameter::clone(AbstractNode *parent_) const {
  return std::unique_ptr<FunctionParameter>(clone_impl(parent_));
}

std::string FunctionParameter::getIdentifier() const {
  return identifier;
}

const Datatype &FunctionParameter::getParameterType() const {
  return parameter_type;
}

Datatype &FunctionParameter::getParameterType() {
  return parameter_type;
}

///////////////////////////////////////////////
////////// AbstractNode Interface /////////////
///////////////////////////////////////////////
FunctionParameter *FunctionParameter::clone_impl(AbstractNode *parent_) const {
  auto p = new FunctionParameter(parameter_type, identifier);
  if (parent_) { p->setParent(*parent_); }
  return p;
}

void FunctionParameter::accept(IVisitor &v) {
  v.visit(*this);
}

AbstractNode::iterator FunctionParameter::begin() {
  return iterator(std::make_unique<EmptyIteratorImpl<AbstractNode>>(*this));
}

AbstractNode::const_iterator FunctionParameter::begin() const {
  return const_iterator(std::make_unique<EmptyIteratorImpl<const AbstractNode>>(*this));
}

AbstractNode::iterator FunctionParameter::end() {
  return iterator(std::make_unique<EmptyIteratorImpl<AbstractNode>>(*this));
}

AbstractNode::const_iterator FunctionParameter::end() const {
  return const_iterator(std::make_unique<EmptyIteratorImpl<const AbstractNode>>(*this));
}

size_t FunctionParameter::countChildren() const {
  return 0;
}

nlohmann::json FunctionParameter::toJson() const {
  nlohmann::json j;
  j["type"] = getNodeType();
  j["parameter_type"] = getParameterType().toString();
  j["identifier"] = getIdentifier();
  return j;
}

std::string FunctionParameter::toString(bool printChildren) const {
  return AbstractNode::toStringHelper(printChildren, {getParameterType().toString(), getIdentifier()});
}

std::string FunctionParameter::getNodeType() const {
  return "FunctionParameter";
}
std::unique_ptr<AbstractNode> FunctionParameter::replaceChild(const AbstractNode &child,
                                                              std::unique_ptr<AbstractNode> &&new_child) {
  throw std::runtime_error("Cannot replace child: This type of node does not have children.");
}
