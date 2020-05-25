#include "ast_opt/ast/Rotate.h"

Rotate::Rotate(AbstractExpr *vector, AbstractExpr *rotationFactor) {
  setAttributes(vector, rotationFactor);
}

Rotate::Rotate(AbstractExpr *vector, int rotationFactor) {
  setAttributes(vector, new LiteralInt(rotationFactor));
}

Rotate::Rotate() = default;

AbstractExpr *Rotate::getRotationFactor() const {
  return dynamic_cast<AbstractExpr *>(getChildAtIndex(1));
}

int Rotate::getMaxNumberChildren() {
  return 2;
}

json Rotate::toJson() const {
  json j;
  j["type"] = getNodeType();
  j["operand"] = getOperand()->toJson();
  j["rotationFactor"] = getRotationFactor()->toJson();
  return j;
}

std::string Rotate::toString(bool printChildren) const {
  return AbstractNode::generateOutputString(printChildren, {});
}

AbstractNode *Rotate::cloneFlat() {
  return new Rotate();
}

bool Rotate::supportsCircuitMode() {
  return true;
}

AbstractExpr *Rotate::getOperand() const {
  return dynamic_cast<AbstractExpr *>(getChildAtIndex(0));
}

std::string Rotate::getNodeType() const {
  return std::string("Rotate");
}

void Rotate::accept(Visitor &v) {
  v.visit(*this);
}

Rotate *Rotate::clone(bool keepOriginalUniqueNodeId) const {
  return new Rotate(getOperand()->clone(keepOriginalUniqueNodeId)->castTo<AbstractExpr>(),
                    getRotationFactor()->clone(keepOriginalUniqueNodeId)->castTo<AbstractExpr>());
}

void Rotate::setAttributes(AbstractExpr *pExpr, AbstractExpr *rotationFactor) {
  // Rotation requires either an AbstractLiteral that is a 1-dimensional row or column vector, or a Variable in which
  // case it is not possible at compile-time to determine whether the variable satisfies the former requirement. Must
  // be checked while evaluating the AST.
  auto pExprAsLiteral = dynamic_cast<AbstractLiteral *>(pExpr);
  if (pExprAsLiteral!=nullptr && !pExprAsLiteral->getMatrix()->containsAbstractExprs()
      && !isOneDimensionalVector(pExpr)) {
    throw std::logic_error("Rotate requires a 1-dimensional row or column vector.");
  }
  removeChildren();
  addChildren({pExpr, rotationFactor}, true);
}

bool Rotate::isOneDimensionalVector(AbstractExpr *operand) {
  if (auto literal = dynamic_cast<AbstractLiteral *>(operand)) {
    auto dim = literal->getMatrix()->getDimensions();
    return dim.equals(1, -1) || dim.equals(-1, 1);
  }
  return false;
}
std::vector<std::string> Rotate::getVariableIdentifiers() {
  auto results = getOperand()->getVariableIdentifiers();
  auto vec = getRotationFactor()->getVariableIdentifiers();
  if (!vec.empty()) {
    results.insert(results.end(), vec.begin(), vec.end());
  }
  return results;
}
std::vector<Variable *> Rotate::getVariables() {
  auto results = getOperand()->getVariables();
  auto vec = getRotationFactor()->getVariables();
  if (!vec.empty()) {
    results.insert(results.end(), vec.begin(), vec.end());
  }
  return results;
}

bool Rotate::isEqual(AbstractExpr *other) {
  if (auto otherAsRotate = dynamic_cast<Rotate *>(other)) {
    return getOperand()->isEqual(otherAsRotate->getOperand())
        && getRotationFactor()->isEqual(otherAsRotate->getRotationFactor());
  }
  return false;
}
