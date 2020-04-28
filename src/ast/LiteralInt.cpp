#include <iostream>
#include "ast_opt/utilities/RandNumGen.h"
#include "ast_opt/ast/LiteralInt.h"
#include "ast_opt/ast/AbstractExpr.h"
#include "ast_opt/ast/Matrix.h"

LiteralInt::LiteralInt(Matrix<AbstractExpr *> *am) : AbstractLiteral(am) {}

LiteralInt::LiteralInt(AbstractMatrix *am) : AbstractLiteral(am) {}

LiteralInt::LiteralInt(int value) : AbstractLiteral(new Matrix(value)) {}

LiteralInt::LiteralInt() : AbstractLiteral(new Matrix<AbstractExpr *>()) {}

LiteralInt::LiteralInt(Matrix<int> *inputMatrix) : AbstractLiteral(inputMatrix) {}

json LiteralInt::toJson() const {
  json j;
  j["type"] = getNodeType();
  j["value"] = matrix->toJson();
  return j;
}

bool LiteralInt::supportsDatatype(Datatype &datatype) {
  return datatype.getType()==Types::INT;
}

int LiteralInt::getValue() const {
  return (*dynamic_cast<Matrix<int> *>(matrix)).getScalarValue();
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
  return *matrix==*rhs.getMatrix();
}

bool LiteralInt::operator!=(const LiteralInt &rhs) const {
  return !(rhs==*this);
}

void LiteralInt::addLiteralValue(std::string identifier,
                                 std::unordered_map<std::string, AbstractLiteral *> &paramsMap) {
  paramsMap.emplace(identifier, this);
}

void LiteralInt::setValue(int newValue) {
  dynamic_cast<Matrix<int> *>(matrix)->setValues({{newValue}});
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
  auto clonedNode = new LiteralInt(matrix->clone(keepOriginalUniqueNodeId));
  clonedNode->updateClone(keepOriginalUniqueNodeId, this);
  return clonedNode;
}

bool LiteralInt::isEqual(AbstractExpr *other) {
  auto otherLiteralInt = dynamic_cast<LiteralInt *>(other);
  return otherLiteralInt!=nullptr && *this==*otherLiteralInt;
}

bool LiteralInt::isNull() {
  auto castedMatrix = dynamic_cast<Matrix<int> *>(matrix);
  return castedMatrix!=nullptr && castedMatrix->allValuesEqual(0);
}

AbstractMatrix *LiteralInt::getMatrix() const {
  return matrix;
}

void LiteralInt::setMatrix(AbstractMatrix *newValue) {
  this->matrix = newValue;
}

