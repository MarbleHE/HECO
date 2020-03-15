#include "LiteralFloat.h"
#include <iostream>
#include "Datatype.h"
#include "AbstractExpr.h"
#include "RandNumGen.h"
#include "Matrix.h"

LiteralFloat::LiteralFloat(Matrix<AbstractExpr *> *am) : AbstractLiteral(am) {}

LiteralFloat::LiteralFloat(AbstractMatrix *am) : AbstractLiteral(am) {}

LiteralFloat::LiteralFloat(float value) : AbstractLiteral(new Matrix(value)) {}

LiteralFloat::LiteralFloat(Matrix<float> *inputMatrix) : AbstractLiteral(inputMatrix) {}

json LiteralFloat::toJson() const {
  json j;
  j["type"] = getNodeType();
  j["value"] = matrix->toJson();
  return j;
}

float LiteralFloat::getValue() const {
  return (*dynamic_cast<Matrix<float> *>(matrix)).getScalarValue();
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
  return *matrix==*rhs.getMatrix();
}

bool LiteralFloat::operator!=(const LiteralFloat &rhs) const {
  return !(rhs==*this);
}

void LiteralFloat::addLiteralValue(std::string identifier,
                                   std::unordered_map<std::string, AbstractLiteral *> &paramsMap) {
  paramsMap.emplace(identifier, this);
}

void LiteralFloat::setValue(float newValue) {
  dynamic_cast<Matrix<float> *>(matrix)->setValues({{newValue}});
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
  auto clonedNode = new LiteralFloat(matrix->clone(keepOriginalUniqueNodeId));
  clonedNode->updateClone(keepOriginalUniqueNodeId, this);
  return clonedNode;
}

bool LiteralFloat::isEqual(AbstractExpr *other) {
  auto otherLiteralFloat = dynamic_cast<LiteralFloat *>(other);
  return otherLiteralFloat!=nullptr && *this==*otherLiteralFloat;
}

bool LiteralFloat::isNull() {
  auto castedMatrix = dynamic_cast<Matrix<float> *>(matrix);
  return castedMatrix!=nullptr && castedMatrix->allValuesEqual(0.0f);
}

AbstractMatrix *LiteralFloat::getMatrix() const {
  return matrix;
}

void LiteralFloat::setMatrix(AbstractMatrix *newValue) {
  this->matrix = newValue;
}
