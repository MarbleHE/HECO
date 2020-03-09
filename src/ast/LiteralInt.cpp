#include "LiteralInt.h"
#include <iostream>
#include "RandNumGen.h"
#include "AbstractExpr.h"

bool LiteralInt::supportsDatatype(Datatype &datatype) {
  return datatype.getType()==Types::INT;
}

LiteralInt::LiteralInt(int value) : matrix(new Matrix(value)) {}

LiteralInt::LiteralInt(Matrix<int> *inputMatrix) : matrix(inputMatrix) {}

json LiteralInt::toJson() const {
  json j;
  j["type"] = getNodeType();
  j["value"] = matrix->toJson();
  return j;
}

int LiteralInt::getValue() const {
  return matrix->getScalarValue();
}

void LiteralInt::accept(Visitor &v) {
  v.visit(*this);
}

std::string LiteralInt::getNodeType() const {
  return "LiteralInt";
}

LiteralInt::~LiteralInt() = default;

void LiteralInt::print(std::ostream &str) const {
  str << matrix->toString();
}

bool LiteralInt::operator==(const LiteralInt &rhs) const {
  return *matrix==*dynamic_cast<Matrix<int> *>(rhs.getMatrix());
}

bool LiteralInt::operator!=(const LiteralInt &rhs) const {
  return !(rhs==*this);
}

void LiteralInt::addLiteralValue(std::string identifier,
                                 std::unordered_map<std::string, AbstractLiteral *> &paramsMap) {
  paramsMap.emplace(identifier, this);
}

void LiteralInt::setValue(int newValue) {
  matrix->setValues({{newValue}});
}

void LiteralInt::setRandomValue(RandLiteralGen &rlg) {
  setValue(rlg.getRandomInt());
}

std::string LiteralInt::toString(bool printChildren) const {
  return AbstractNode::generateOutputString(printChildren, {matrix->toString()});
}

bool LiteralInt::supportsCircuitMode() {
  return true;
}

LiteralInt *LiteralInt::clone(bool keepOriginalUniqueNodeId) {
  auto clonedNode = new LiteralInt(matrix->clone());
  if (keepOriginalUniqueNodeId) clonedNode->setUniqueNodeId(this->getUniqueNodeId());
  if (this->isReversed) clonedNode->swapChildrenParents();
  return clonedNode;
}

bool LiteralInt::isEqual(AbstractExpr *other) {
  auto otherLiteralInt = dynamic_cast<LiteralInt *>(other);
  return otherLiteralInt!=nullptr && *this==*otherLiteralInt;
}

bool LiteralInt::isNull() {
  return matrix->allValuesEqual(0);
}
AbstractMatrix *LiteralInt::getMatrix() const {
  return matrix;
}
