#include "ast_opt/ast/Rotate.h"

Rotate::Rotate(AbstractExpr *vector, AbstractExpr *rotationFactor) {
  setAttributes(vector, rotationFactor);
}

Rotate::Rotate(AbstractExpr *vector, int rotationFactor) {
  setAttributes(vector, new LiteralInt(rotationFactor));
}

Rotate::Rotate() = default;

AbstractExpr *Rotate::getRotationFactor() const {
  return reinterpret_cast<AbstractExpr *>(getChildAtIndex(1));
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
  return reinterpret_cast<AbstractExpr *>(getChildAtIndex(0));
}

std::string Rotate::getNodeType() const {
  return std::string("Rotate");
}

void Rotate::accept(Visitor &v) {
  v.visit(*this);
}

Rotate *Rotate::clone(bool keepOriginalUniqueNodeId) {
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
  Dimension *dim;
  if (auto literal = dynamic_cast<AbstractLiteral *>(operand)) {
    dim = &(literal->getMatrix()->getDimensions());
    return dim->equals(1, -1) || dim->equals(-1, 1);
  }
  return false;
}
