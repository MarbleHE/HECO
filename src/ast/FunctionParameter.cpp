#include <utility>
#include "ast_opt/ast/FunctionParameter.h"
#include "ast_opt/visitor/IVisitor.h"

FunctionParameter::~FunctionParameter() = default;

FunctionParameter::FunctionParameter(std::string identifier, Datatype parameter_type)
    : identifier(std::move(identifier)), parameter_type(std::move(parameter_type)) {}

FunctionParameter::FunctionParameter(const FunctionParameter &other) : identifier(other.identifier), parameter_type(other.parameter_type){}

FunctionParameter::FunctionParameter(FunctionParameter &&other) noexcept: identifier(std::move(other.identifier)), parameter_type(std::move(other.parameter_type)) {}

FunctionParameter &FunctionParameter::operator=(const FunctionParameter &other) {
  identifier = other.identifier;
  return *this;
}
FunctionParameter &FunctionParameter::operator=(FunctionParameter &&other) noexcept {
  identifier = std::move(other.identifier);
  return *this;
}

std::unique_ptr<FunctionParameter> FunctionParameter::clone() const {
  return std::unique_ptr<FunctionParameter>(clone_impl());
}

std::string FunctionParameter::getIdentifier() const {
  return identifier;
}

Datatype FunctionParameter::getParameterType() const {
  return parameter_type;
}


///////////////////////////////////////////////
////////// AbstractNode Interface /////////////
///////////////////////////////////////////////
FunctionParameter *FunctionParameter::clone_impl() const {
  return new FunctionParameter(identifier, parameter_type);
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