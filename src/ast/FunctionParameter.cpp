#include "FunctionParameter.h"

void to_json(json &j, const FunctionParameter &funcParam) {
  j = {
      {"type", funcParam.getNodeName()},
      {"value", funcParam.getValue()->toJson()},
      {"datatype", funcParam.getDatatype()->toString()}
  };
}

void to_json(json &j, const FunctionParameter* funcParam) {
  j = {
      {"type", funcParam->getNodeName()},
      {"value", funcParam->getValue()->toJson()},
      {"datatype", funcParam->getDatatype()->toString()}
  };
}

json FunctionParameter::toJson() const {
  json j = {
      {"type", getNodeName()},
      {"value", getValue()->toJson()},
      {"datatype", getDatatype()->toString()}
  };
  return j;
}

void FunctionParameter::accept(Visitor &v) {
  v.visit(*this);
}

std::string FunctionParameter::getNodeName() const {
  return "FunctionParameter";
}

AbstractExpr* FunctionParameter::getValue() const {
  return reinterpret_cast<AbstractExpr* >(getChildAtIndex(1, true));
}

FunctionParameter::FunctionParameter(Datatype* datatype, AbstractExpr* value) {
  setAttributes(datatype, value);
}

FunctionParameter::FunctionParameter(const std::string &datatypeEnumString, AbstractExpr* value) {
  setAttributes(new Datatype(datatypeEnumString), value);
}

Datatype* FunctionParameter::getDatatype() const {
  return reinterpret_cast<Datatype* >(getChildAtIndex(0, true));
}

Node* FunctionParameter::createClonedNode(bool keepOriginalUniqueNodeId) {
  return new FunctionParameter(this->getDatatype()->cloneRecursiveDeep(keepOriginalUniqueNodeId)->castTo<Datatype>(),
                               this->getValue()->cloneRecursiveDeep(keepOriginalUniqueNodeId)->castTo<AbstractExpr>());
}

int FunctionParameter::getMaxNumberChildren() {
  return 2;
}

bool FunctionParameter::supportsCircuitMode() {
  return true;
}

void FunctionParameter::setAttributes(Datatype* datatype, AbstractExpr* value) {
  // update tree structure
  removeChildren();
  addChildren({datatype, value}, false);
  Node::addParentTo(this, {datatype, value});
}

bool FunctionParameter::operator==(const FunctionParameter &rhs) const {
  return getValue() == rhs.getValue() && getDatatype()->getType() == rhs.getDatatype()->getType();
}

bool FunctionParameter::operator!=(const FunctionParameter &rhs) const {
  return !(rhs == *this);
}

