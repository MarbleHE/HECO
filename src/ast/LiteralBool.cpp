#include "LiteralBool.h"
#include "RandNumGen.h"
#include "Matrix.h"

LiteralBool::LiteralBool(Matrix<AbstractExpr *> *am) : AbstractLiteral(am) {}

LiteralBool::LiteralBool(AbstractMatrix *am) : AbstractLiteral(am) {}

LiteralBool::LiteralBool(bool value) : AbstractLiteral(new Matrix(value)) {}

LiteralBool::LiteralBool() : AbstractLiteral(new Matrix<AbstractExpr *>()) {}

LiteralBool::LiteralBool(Matrix<bool> *inputMatrix) : AbstractLiteral(inputMatrix) {}

json LiteralBool::toJson() const {
  json j;
  j["type"] = getNodeType();
  j["value"] = matrix->toJson();
  return j;
}

void LiteralBool::accept(Visitor &v) {
  v.visit(*this);
}

bool LiteralBool::getValue() const {
  return (*dynamic_cast<Matrix<bool> *>(matrix)).getScalarValue();
}

std::string LiteralBool::getNodeType() const {
  return "LiteralBool";
}

LiteralBool::~LiteralBool() = default;

void LiteralBool::print(std::ostream &str) const {
  str << matrix->toString();
}

bool LiteralBool::operator==(const LiteralBool &rhs) const {
  return *matrix==*rhs.getMatrix();
}

bool LiteralBool::operator!=(const LiteralBool &rhs) const {
  return !(rhs==*this);
}

void LiteralBool::addLiteralValue(std::string identifier,
                                  std::unordered_map<std::string, AbstractLiteral *> &paramsMap) {
  paramsMap.emplace(identifier, this);
}

void LiteralBool::setValue(bool newValue) {
  dynamic_cast<Matrix<bool> *>(matrix)->setValues({{newValue}});
}

void LiteralBool::setRandomValue(RandLiteralGen &rlg) {
  setValue(rlg.getRandomBool());
}

std::string LiteralBool::toString(bool printChildren) const {
  return AbstractNode::generateOutputString(printChildren, {matrix->toString()});
}

bool LiteralBool::supportsCircuitMode() {
  return true;
}

bool LiteralBool::supportsDatatype(Datatype &datatype) {
  return datatype.getType()==Types::BOOL;
}

LiteralBool *LiteralBool::clone(bool keepOriginalUniqueNodeId) {
  auto clonedNode = new LiteralBool(matrix->clone(keepOriginalUniqueNodeId));
  clonedNode->updateClone(keepOriginalUniqueNodeId, this);
  return clonedNode;
}

bool LiteralBool::isEqual(AbstractExpr *other) {
  auto otherLiteralBool = dynamic_cast<LiteralBool *>(other);
  return otherLiteralBool!=nullptr && *this==*otherLiteralBool;
}

bool LiteralBool::isNull() {
  auto castedMatrix = dynamic_cast<Matrix<bool> *>(matrix);
  return castedMatrix!=nullptr && castedMatrix->allValuesEqual(false);
}

AbstractMatrix *LiteralBool::getMatrix() const {
  return matrix;
}

void LiteralBool::setMatrix(AbstractMatrix *newValue) {
  this->matrix = dynamic_cast<Matrix<bool> *>(newValue);
}
