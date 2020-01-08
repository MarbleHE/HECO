
#include <utility>
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
  return value;
}

FunctionParameter::FunctionParameter(Datatype* datatype, AbstractExpr* value) : datatype(datatype), value(value) {}

FunctionParameter::FunctionParameter(std::string datatype, AbstractExpr* value) : value(value) {
  this->datatype = new Datatype(std::move(datatype));
}

Datatype* FunctionParameter::getDatatype() const {
  return datatype;
}

