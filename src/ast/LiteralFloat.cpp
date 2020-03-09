#include "LiteralFloat.h"
#include <iostream>
#include "Datatype.h"
#include "AbstractExpr.h"
#include "RandNumGen.h"

LiteralFloat::LiteralFloat(float value) : matrix(new Matrix(value)) {}

LiteralFloat::LiteralFloat(Matrix<float> *inputMatrix) : matrix(inputMatrix) {}

json LiteralFloat::toJson() const {
  json j;
  j["type"] = getNodeType();
  j["value"] = matrix->toJson();
  return j;
}

float LiteralFloat::getValue() const {
  return matrix->getScalarValue();
}

void LiteralFloat::accept(Visitor &v) {
  v.visit(*this);
}

std::string LiteralFloat::getNodeType() const {
  return "LiteralFloat";
}

LiteralFloat::~LiteralFloat() = default;

void LiteralFloat::print(std::ostream &str) const {
  str << matrix->toString();
}

bool LiteralFloat::operator==(const LiteralFloat &rhs) const {
  return *matrix==*dynamic_cast<Matrix<float> *>(rhs.getMatrix());
}

bool LiteralFloat::operator!=(const LiteralFloat &rhs) const {
  return !(rhs==*this);
}

void LiteralFloat::addLiteralValue(std::string identifier,
                                   std::unordered_map<std::string, AbstractLiteral *> &paramsMap) {
  paramsMap.emplace(identifier, this);
}

void LiteralFloat::setValue(float newValue) {
  matrix->setValues({{newValue}});
}

void LiteralFloat::setRandomValue(RandLiteralGen &rlg) {
  setValue(rlg.getRandomFloat());
}

std::string LiteralFloat::toString(bool printChildren) const {
  return AbstractNode::generateOutputString(printChildren, {matrix->toString()});

}

bool LiteralFloat::supportsCircuitMode() {
  return true;
}

bool LiteralFloat::supportsDatatype(Datatype &datatype) {
  return datatype.getType()==Types::FLOAT;
}

LiteralFloat *LiteralFloat::clone(bool keepOriginalUniqueNodeId) {
  auto clonedNode = new LiteralFloat(matrix->clone());
  if (keepOriginalUniqueNodeId) clonedNode->setUniqueNodeId(this->getUniqueNodeId());
  if (this->isReversed) clonedNode->swapChildrenParents();
  return clonedNode;
}

bool LiteralFloat::isEqual(AbstractExpr *other) {
  auto otherLiteralFloat = dynamic_cast<LiteralFloat *>(other);
  return otherLiteralFloat!=nullptr && *this==*otherLiteralFloat;
}

bool LiteralFloat::isNull() {
  return matrix->allValuesEqual(0.0f);
}
CMatrix *LiteralFloat::getMatrix() const {
  return matrix;
}
