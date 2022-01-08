#include <unordered_map>
#include "ast_opt/ast_parser/Errors.h"
#include "ast_opt/ast_utilities/Datatype.h"

std::string enumToString(const Type type) {
  std::unordered_map<Type, std::string> typeToString = {
      {Type::BOOL, "bool"},
      {Type::CHAR, "char"},
      {Type::INT, "int"},
      {Type::FLOAT, "float"},
      {Type::DOUBLE, "double"},
      {Type::STRING, "string"},
      {Type::VOID, "void"}
  };
  return typeToString.find(type)->second;
}

Type stringToTypeEnum(const std::string s) {
  for ( auto type : all_types) {
    if (enumToString(type)  == s)
      return type;
  }
  throw stork::runtime_error("No value type Type found for '" + s + "'!");
}

const std::string Datatype::secretPrefix = "secret ";

Datatype::Datatype(Type type, bool isSecret) : type(type), isSecret(isSecret) {}

Datatype::Datatype(std::string typeString) {
  if (typeString.rfind(secretPrefix, 0) == 0) {
    // remove secret prefix
    this->type = stringToTypeEnum(typeString.substr(secretPrefix.size(), typeString.size() - secretPrefix.size()));
    this->isSecret = true;
  }
  else {
    this->type = stringToTypeEnum(typeString);
    this->isSecret = false;
  }
}

bool Datatype::operator==(const Datatype &rhs) const {
  return type==rhs.type && isSecret==rhs.isSecret;
}

bool Datatype::operator!=(const Datatype &rhs) const {
  return !(*this==rhs);
}

Type Datatype::getType() const {
  return type;
}

bool Datatype::getSecretFlag() const {
  return isSecret;
}

std::string Datatype::toString() const {
  return isSecret ? secretPrefix + enumToString(type) : enumToString(type);
}
