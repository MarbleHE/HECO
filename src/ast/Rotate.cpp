#include "Rotate.h"

Rotate::Rotate(AbstractExpr *vector, int rotationFactor) : rotationFactor(rotationFactor) {
  setAttributes(vector);
}

int Rotate::getRotationFactor() const {
  return rotationFactor;
}

int Rotate::getMaxNumberChildren() {
  return 1;
}

json Rotate::toJson() const {
  json j;
  j["type"] = getNodeType();
  j["operand"] = getOperand()->toJson();
  j["rotationFactor"] = getRotationFactor();
  return j;
}

std::string Rotate::toString(bool printChildren) const {
  return AbstractNode::generateOutputString(printChildren, {std::to_string(getRotationFactor())});
}

AbstractNode *Rotate::cloneFlat() {
  return new Rotate(nullptr, this->getRotationFactor());
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
  return new Rotate(this->getChildAtIndex(0)->clone(keepOriginalUniqueNodeId)->castTo<AbstractExpr>(),
                    getRotationFactor());
}

void Rotate::setAttributes(AbstractExpr *pExpr) {
  // Rotation requires either an AbstractLiteral that is a 1-dimensional row or column vector, or a Variable in which
  // case it is not possible at compile-time to determine whether the variable satisfies the former requirement. Must
  // be checked while evaluating the AST.
  if (dynamic_cast<AbstractLiteral *>(pExpr)!=nullptr && !isOneDimensionalVector()) {
    throw std::logic_error("Rotate requires a 1-dimensional row or column vector.");
  } else if (dynamic_cast<Variable *>(pExpr)==nullptr) {
    throw std::logic_error("Rotate is supported for AbstractLiterals and Variables only.");
  }
  removeChildren();
  addChildren({pExpr}, true);
}

bool Rotate::isOneDimensionalVector() {
  Dimension *dim;
  auto expressionToRotate = getOperand();
  if (auto literal = dynamic_cast<AbstractLiteral *>(expressionToRotate)) {
    dim = &(literal->getMatrix()->getDimensions());
    return dim->equals(1, -1) || dim->equals(-1, 1);
  }
  return false;
}
