#include "Variable.h"
#include <utility>
#include "Ast.h"

Variable::Variable(Matrix<std::string> *inputMatrix) : matrix(inputMatrix) {}

Variable::Variable(std::string identifier) : matrix(new Matrix(std::move(identifier))) {}

json Variable::toJson() const {
  json j;
  j["type"] = getNodeType();
  j["identifier"] = matrix->toJson();
  return j;
}

void Variable::accept(Visitor &v) {
  v.visit(*this);
}

std::string Variable::getNodeType() const {
  return "Variable";
}

std::string Variable::getIdentifier() const {
  return matrix->getScalarValue();
}

bool Variable::contains(Variable *var) {
  return *this==*var;
}

bool Variable::operator==(const Variable &rhs) const {
  return *matrix==*rhs.getMatrix();
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
  return matrix->getValuesAsVector();
}

std::string Variable::toString(bool printChildren) const {
  return AbstractNode::generateOutputString(printChildren, {matrix->toString()});
}

bool Variable::supportsCircuitMode() {
  return true;
}

Variable::~Variable() = default;

Variable *Variable::clone(bool keepOriginalUniqueNodeId) {
  auto clonedNode = new Variable(matrix->clone());
  if (keepOriginalUniqueNodeId) clonedNode->setUniqueNodeId(this->getUniqueNodeId());
  if (this->isReversed) clonedNode->swapChildrenParents();
  return clonedNode;
}

Matrix<std::string> *Variable::getMatrix() const {
  return matrix;
}
