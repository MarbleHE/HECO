#include <utility>
#include "ast_opt/ast/Datatype.h"
#include "ast_opt/ast/LiteralString.h"
#include "ast_opt/ast/Matrix.h"
#include "ast_opt/utilities/RandNumGen.h"

LiteralString::LiteralString(Matrix<AbstractExpr *> *am) : AbstractLiteral(am) {}

LiteralString::LiteralString(AbstractMatrix *pMatrix) : AbstractLiteral(pMatrix) {}

LiteralString::LiteralString(Matrix<std::string> *inputMatrix) : AbstractLiteral(inputMatrix) {}

LiteralString::LiteralString(std::string value) : AbstractLiteral(new Matrix(std::move(value))) {}

LiteralString::LiteralString() : AbstractLiteral(new Matrix<AbstractExpr *>()) {}

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
  return (*dynamic_cast<Matrix<std::string> *>(matrix)).getScalarValue();
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
  dynamic_cast<Matrix<std::string> *>(matrix)->setValues({{newValue}});
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

LiteralString *LiteralString::clone(bool keepOriginalUniqueNodeId) const {
  auto clonedNode = new LiteralString(matrix->clone(keepOriginalUniqueNodeId));
  clonedNode->updateClone(keepOriginalUniqueNodeId, this);
  return clonedNode;
}

bool LiteralString::isEqual(AbstractExpr *other) {
  auto otherLiteralString = dynamic_cast<LiteralString *>(other);
  return otherLiteralString!=nullptr && *this==*otherLiteralString;
}

bool LiteralString::isNull() {
  auto castedMatrix = dynamic_cast<Matrix<std::string> *>(matrix);
  return castedMatrix!=nullptr && castedMatrix->allValuesEqual(std::string(""));
}

AbstractMatrix *LiteralString::getMatrix() const {
  return matrix;
}

void LiteralString::setMatrix(AbstractMatrix *newValue) {
  this->matrix = dynamic_cast<Matrix<std::string> *>(newValue);
}
