#include <utility>
#include "heco/ast/Variable.h"
#include "heco/ast_utilities/IVisitor.h"

Variable::~Variable() = default;

Variable::Variable(std::string variableIdentifier) : identifier(std::move(variableIdentifier)) {}

Variable::Variable(const Variable &other) : identifier(other.identifier) {}

Variable::Variable(Variable &&other) noexcept : identifier(std::move(other.identifier)) {}

Variable &Variable::operator=(const Variable &other)
{
  identifier = other.identifier;
  return *this;
}
Variable &Variable::operator=(Variable &&other) noexcept
{
  identifier = std::move(other.identifier);
  return *this;
}

std::unique_ptr<Variable> Variable::clone(AbstractNode *parent_) const
{
  return std::unique_ptr<Variable>(clone_impl(parent_));
}

std::string Variable::getIdentifier() const
{
  return identifier;
}

///////////////////////////////////////////////
////////// AbstractNode Interface /////////////
///////////////////////////////////////////////
Variable *Variable::clone_impl(AbstractNode *parent_) const
{
  auto p = new Variable(identifier);
  if (parent_)
  {
    p->setParent(*parent_);
  }
  return p;
}

void Variable::accept(IVisitor &v)
{
  v.visit(*this);
}

AbstractNode::iterator Variable::begin()
{
  return iterator(std::make_unique<EmptyIteratorImpl<AbstractNode>>(*this));
}

AbstractNode::const_iterator Variable::begin() const
{
  return const_iterator(std::make_unique<EmptyIteratorImpl<const AbstractNode>>(*this));
}

AbstractNode::iterator Variable::end()
{
  return iterator(std::make_unique<EmptyIteratorImpl<AbstractNode>>(*this));
}

AbstractNode::const_iterator Variable::end() const
{
  return const_iterator(std::make_unique<EmptyIteratorImpl<const AbstractNode>>(*this));
}

size_t Variable::countChildren() const
{
  return 0;
}

nlohmann::json Variable::toJson() const
{
  nlohmann::json j;
  j["type"] = getNodeType();
  j["identifier"] = getIdentifier();
  return j;
}

std::unique_ptr<Variable> Variable::fromJson(nlohmann::json j)
{
  return std::make_unique<Variable>(j["identifier"].get<std::string>());
}

std::string Variable::toString(bool printChildren) const
{
  return AbstractNode::toStringHelper(printChildren, {getIdentifier()});
}

std::string Variable::getNodeType() const
{
  return "Variable";
}