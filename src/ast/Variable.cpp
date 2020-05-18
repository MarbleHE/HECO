#include <utility>
#include "ast_opt/ast/Variable.h"
#include "ast_opt/ast/Ast.h"

Variable::Variable(std::string variableIdentifier) : identifier(std::move(variableIdentifier)) {}

json Variable::toJson() const {
  json j;
  j["type"] = getNodeType();
  j["identifier"] = getIdentifier();
  return j;
}

void Variable::accept(Visitor &v) {
  v.visit(*this);
}

std::string Variable::getNodeType() const {
  return "Variable";
}

std::string Variable::getIdentifier() const {
  return identifier;
}

bool Variable::contains(Variable *var) {
  return *this==*var;
}

bool Variable::operator==(const Variable &rhs) const {
  return identifier==rhs.getIdentifier();
}

bool Variable::operator!=(const Variable &rhs) const {
  return !(rhs==*this);
}

bool Variable::isEqual(AbstractExpr *other) {
  if (auto otherVar = dynamic_cast<Variable *>(other)) {
    return *this==*otherVar;
  }
  return false;
}

std::vector<std::string> Variable::getVariableIdentifiers() {
  return {getIdentifier()};
}

std::vector<Variable *> Variable::getVariables() {
  return {this};
}

std::string Variable::toString(bool printChildren) const {
  return AbstractNode::generateOutputString(printChildren, {getIdentifier()});
}

bool Variable::supportsCircuitMode() {
  return true;
}

Variable::~Variable() = default;

Variable *Variable::clone(bool keepOriginalUniqueNodeId) const {
  auto clonedNode = new Variable(getIdentifier());
  clonedNode->updateClone(keepOriginalUniqueNodeId, this);
  return clonedNode;
}
