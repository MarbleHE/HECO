#include <utility>
#include <Datatypes.h>

#include "../../include/ast/LiteralString.h"
#include "../utilities/RandNumGen.h"

LiteralString::LiteralString(std::string value) : value(std::move(value)) {}

json LiteralString::toJson() const {
  json j;
  j["type"] = getNodeName();
  j["value"] = this->value;
  return j;
}

void LiteralString::accept(Visitor &v) {
  v.visit(*this);
}

const std::string &LiteralString::getValue() const {
  return value;
}

std::string LiteralString::getNodeName() const {
  return "LiteralString";
}

LiteralString::~LiteralString() = default;

Literal* LiteralString::evaluate(Ast &ast) {
  return this;
}

void LiteralString::print(std::ostream &str) const {
  str << this->getValue();
}

bool LiteralString::operator==(const LiteralString &rhs) const {
  return value == rhs.value;
}

bool LiteralString::operator!=(const LiteralString &rhs) const {
  return !(rhs == *this);
}

void LiteralString::addLiteralValue(std::string identifier, std::map<std::string, Literal*> &paramsMap) {
  paramsMap.emplace(identifier, this);
}

void LiteralString::setValue(const std::string &newValue) {
  this->value = newValue;
}

void LiteralString::setRandomValue(RandLiteralGen &rlg) {
  setValue(rlg.getRandomString(RandLiteralGen::stringDefaultMaxLength));
}

std::string LiteralString::toString() const {
  return this->getValue();
}

bool LiteralString::supportsCircuitMode() {
  return true;
}

bool LiteralString::supportsDatatype(Datatype &datatype) {
  return datatype.getType() == TYPES::STRING;
}
