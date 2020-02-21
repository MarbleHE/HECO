#include <iostream>
#include "Datatype.h"
#include "AbstractExpr.h"
#include "LiteralFloat.h"
#include "RandNumGen.h"

LiteralFloat::LiteralFloat(float value) : value(value) {}

json LiteralFloat::toJson() const {
  json j;
  j["type"] = getNodeName();
  j["value"] = this->value;
  return j;
}

float LiteralFloat::getValue() const {
  return value;
}

void LiteralFloat::accept(Visitor &v) {
  v.visit(*this);
}

std::string LiteralFloat::getNodeName() const {
  return "LiteralFloat";
}

LiteralFloat::~LiteralFloat() = default;

LiteralFloat LiteralFloat::operator+(LiteralFloat const &lint) {
  return LiteralFloat(this->getValue() + lint.getValue());
}

std::ostream &operator<<(std::ostream &os, const LiteralFloat &an_int) {
  os << an_int.getValue();
  return os;
}

void LiteralFloat::print(std::ostream &str) const {
  str << this->value;
}

bool LiteralFloat::operator==(const LiteralFloat &rhs) const {
  return value==rhs.value;
}

bool LiteralFloat::operator!=(const LiteralFloat &rhs) const {
  return !(rhs==*this);
}

void LiteralFloat::addLiteralValue(std::string identifier,
                                   std::unordered_map<std::string, AbstractLiteral *> &paramsMap) {
  paramsMap.emplace(identifier, this);
}

void LiteralFloat::setValue(float newValue) {
  this->value = newValue;
}

void LiteralFloat::setRandomValue(RandLiteralGen &rlg) {
  setValue(rlg.getRandomFloat());
}

std::string LiteralFloat::toString() const {
  return std::to_string(this->getValue());
}

bool LiteralFloat::supportsCircuitMode() {
  return true;
}

bool LiteralFloat::supportsDatatype(Datatype &datatype) {
  return datatype.getType()==Types::FLOAT;
}

LiteralFloat *LiteralFloat::clone(bool keepOriginalUniqueNodeId) {
  auto clonedNode = new LiteralFloat(this->getValue());
  if (keepOriginalUniqueNodeId) clonedNode->setUniqueNodeId(this->getUniqueNodeId());
  if (this->isReversed) clonedNode->swapChildrenParents();
  return clonedNode;
}

bool LiteralFloat::isEqual(AbstractExpr *other) {
  auto otherLiteralFloat = dynamic_cast<LiteralFloat *>(other);
  return otherLiteralFloat!=nullptr && this->getValue()==otherLiteralFloat->getValue();
}
