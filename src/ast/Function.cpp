#include <iostream>
#include <exception>
#include <utility>
#include "ast_opt/ast/Function.h"
#include "ast_opt/visitor/IVisitor.h"

/// Convenience typedef for conciseness
typedef std::unique_ptr<AbstractStatement> exprPtr;

Function::~Function() = default;

Function::Function(Datatype return_type,
                   std::string identifier,
                   std::vector<std::unique_ptr<FunctionParameter>> parameters,
                   std::unique_ptr<Block> body)
    : return_type(return_type),
      identifier(std::move(identifier)),
      parameters(std::move(parameters)),
      body(std::move(body)) {}

Function::Function(const Function &other) : return_type(other.return_type),
                                            identifier(other.identifier),
                                            body(other.body->clone()) {
  // deep-copy the parameters, including nullptrs
  parameters.reserve(other.parameters.size());
  for (auto &p: other.parameters) {
    parameters.emplace_back(p ? p->clone() : nullptr);
  }

}

Function::Function(Function &&other) noexcept: return_type(std::move(other.return_type)),
                                               identifier(std::move(other.identifier)),
                                               parameters(std::move(other.parameters)),
                                               body(std::move(other.body)) {}

Function &Function::operator=(const Function &other) {
  return_type = other.return_type;
  identifier = other.identifier;
  // deep-copy the statements, including nullptrs
  parameters.clear();
  parameters.reserve(other.parameters.size());
  for (auto &p: other.parameters) {
    parameters.emplace_back(p ? p->clone() : nullptr);
  }
  body = other.body->clone();
  return *this;
}
Function &Function::operator=(Function &&other) noexcept {
  return_type = std::move(other.return_type);
  identifier = std::move(other.identifier);
  parameters = std::move(other.parameters);
  body = std::move(other.body);
  return *this;
}
std::unique_ptr<Function> Function::clone() const {
  return std::unique_ptr<Function>(clone_impl());
}

Datatype Function::getReturnType() const {
  return return_type;
}
std::string Function::getIdentifier() const {
  return identifier;

}

std::vector<std::reference_wrapper<const FunctionParameter>> Function::getParameters() const {
  std::vector<std::reference_wrapper<const FunctionParameter>> r;
  for (auto &p: parameters) {
    if (p) { r.emplace_back(*p); }
  }
  return r;
}
std::vector<std::reference_wrapper<FunctionParameter>> Function::getParameters() {
  std::vector<std::reference_wrapper<FunctionParameter>> r;
  for (auto &p: parameters) {
    if (p) { r.emplace_back(*p); }
  }
  return r;
}
bool Function::hasBody() const {
  return body!=nullptr;
}
const Block &Function::getBody() const {
  return *body;
}
Block &Function::getBody() {
  return *body;
}
///////////////////////////////////////////////
////////// AbstractNode Interface /////////////
///////////////////////////////////////////////
Function *Function::clone_impl() const {
  return new Function(*this);
}

void Function::accept(IVisitor &v) {
  v.visit(*this);
}

AbstractNode::iterator Function::begin() {
  return AbstractNode::iterator(std::make_unique<FunctionIteratorImpl<AbstractNode>>(*this,
                                                                                     parameters.begin(),
                                                                                     parameters.end(),
                                                                                     0));
}

AbstractNode::const_iterator Function::begin() const {
  return AbstractNode::const_iterator(std::make_unique<FunctionIteratorImpl<const AbstractNode>>(*this,
                                                                                                 parameters.begin(),
                                                                                                 parameters.end(),
                                                                                                 0));
}

AbstractNode::iterator Function::end() {
  return AbstractNode::iterator(std::make_unique<FunctionIteratorImpl<AbstractNode>>(*this,
                                                                                     parameters.end(),
                                                                                     parameters.end(),
                                                                                     1 + hasBody()));
}

AbstractNode::const_iterator Function::end() const {
  return AbstractNode::const_iterator(std::make_unique<FunctionIteratorImpl<const AbstractNode>>(*this,
                                                                                                 parameters.end(),
                                                                                                 parameters.end(),
                                                                                                 1 + hasBody()));
}

size_t Function::countChildren() const {
  // Only non-null entries in the vector are counted as children
  // Because std::unique_ptr doesn't have copy, we can't use std::count_if
  size_t count = 0;
  for (auto &p : parameters) {
    if (p!=nullptr) { count++; }
  }
  count += hasBody();
  return count;
}

nlohmann::json Function::toJson() const {
  std::vector<std::reference_wrapper<const FunctionParameter>> params = getParameters();
  std::vector<nlohmann::json> paramsJson;
  for (const FunctionParameter &p: params) {
    paramsJson.push_back(p.toJson());
  }
  nlohmann::json j = {{"type", getNodeType()},
                      {"return_type", getReturnType().toString()},
                      {"identifier", getIdentifier()},
                      {"parameters", paramsJson}
  };
  if (hasBody()) j["body"] = getBody().toJson();
  return j;
}

std::string Function::toString(bool printChildren) const {
  return AbstractNode::toStringHelper(printChildren, {getReturnType().toString(), getIdentifier()});
}

std::string Function::getNodeType() const {
  return "Function";
}



