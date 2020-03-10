#include <utility>
#include "Datatype.h"
#include "LiteralString.h"
#include "RandNumGen.h"
#include "Matrix.h"

LiteralString::LiteralString(Matrix<std::string> *inputMatrix) : matrix(inputMatrix) {}

LiteralString::LiteralString(std::string value) : matrix(new Matrix(value)) {}

json LiteralString::toJson() const {
  json j;
  j["type"] = getNodeType();
  j["value"] = matrix->toJson();
  return j;
}

void LiteralString::accept(Visitor &v) {
  v.visit(*this);
}

std::string LiteralString::getValue() const {
  return matrix->getScalarValue();
}

std::string LiteralString::getNodeType() const {
  return "LiteralString";
}

LiteralString::~LiteralString() = default;

void LiteralString::print(std::ostream &str) const {
  str << this->getValue();
}

bool LiteralString::operator==(const LiteralString &rhs) const {
  return *matrix==*dynamic_cast<Matrix<std::string> *>(rhs.getMatrix());
}

bool LiteralString::operator!=(const LiteralString &rhs) const {
  return !(rhs==*this);
}

void LiteralString::addLiteralValue(std::string identifier,
                                    std::unordered_map<std::string, AbstractLiteral *> &paramsMap) {
  paramsMap.emplace(identifier, this);
}

void LiteralString::setValue(const std::string &newValue) {
  matrix->setValues({{newValue}});
}

void LiteralString::setRandomValue(RandLiteralGen &rlg) {
  setValue(rlg.getRandomString(RandLiteralGen::stringDefaultMaxLength));
}

std::string LiteralString::toString(bool printChildren) const {
  return AbstractNode::generateOutputString(printChildren, {matrix->toString()});
}

bool LiteralString::supportsCircuitMode() {
  return true;
}

bool LiteralString::supportsDatatype(Datatype &datatype) {
  return datatype.getType()==Types::STRING;
}

LiteralString *LiteralString::clone(bool keepOriginalUniqueNodeId) {
  auto clonedNode = new LiteralString(matrix->clone());
  if (keepOriginalUniqueNodeId) clonedNode->setUniqueNodeId(this->getUniqueNodeId());
  if (this->isReversed) clonedNode->swapChildrenParents();
  return clonedNode;
}

bool LiteralString::isEqual(AbstractExpr *other) {
  auto otherLiteralString = dynamic_cast<LiteralString *>(other);
  return otherLiteralString!=nullptr && *this==*otherLiteralString;
}

bool LiteralString::isNull() {
  return matrix->allValuesEqual("");
}

AbstractMatrix *LiteralString::getMatrix() const {
  return matrix;
}

void LiteralString::setMatrix(AbstractMatrix *newValue) {
  this->matrix = dynamic_cast<Matrix<std::string> *>(newValue);
}
