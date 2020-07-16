#include "ast_opt/utilities/Datatype.h"

std::string enumToString(const Type type) {
  std::unordered_map<Type, std::string> typeToString = {
      {Type::BOOL, "bool"},
      {Type::CHAR, "char"},
      {Type::INT, "int"},
      {Type::FLOAT, "float"},
      {Type::DOUBLE, "double"},
      {Type::STRING, "string"}
  };
  return typeToString.find(type)->second;
}

Datatype::Datatype(Type type, bool isSecret) : type(type), isSecret(isSecret) {}

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
  return isSecret ? "secret + " + enumToString(type) : enumToString(type);
}