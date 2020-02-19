#include "Datatype.h"

Datatype::Datatype(const std::string &type) {
  static const std::unordered_map<std::string, Types> string_to_types = {
      {"int", Types::INT},
      {"float", Types::FLOAT},
      {"string", Types::STRING},
      {"bool", Types::BOOL}};
  auto result = string_to_types.find(type);
  if (result==string_to_types.end()) {
    throw std::invalid_argument(
        "Unsupported datatype given: " + type + ". See the supported datatypes in Datatypes.h.");
  }
  val = result->second;
}

std::string Datatype::enumToString(const Types identifiers) {
  static const std::map<Types, std::string> types_to_string = {
      {Types::INT, "int"},
      {Types::FLOAT, "float"},
      {Types::STRING, "string"},
      {Types::BOOL, "bool"}};
  return types_to_string.find(identifiers)->second;
}

Datatype::operator std::string() const {
  return enumToString(val);
}

Datatype::operator Types() const {
  return val;
}

std::string Datatype::toString() const {
  return enumToString(val);
}

bool Datatype::operator==(const Datatype &rhs) const {
  return val==rhs.val &&
      encrypted==rhs.encrypted;
}

bool Datatype::operator!=(const Datatype &rhs) const {
  return !(rhs==*this);
}

Types Datatype::getType() const {
  return val;
}

bool Datatype::isEncrypted() const {
  return encrypted;
}

void Datatype::setEncrypted(bool encrypted) {
  Datatype::encrypted = encrypted;
}

AbstractNode *Datatype::cloneFlat() {
  //TODO(vianda): Implement cloneFlat for Datatypes
  throw std::runtime_error("oops");
}

AbstractNode *Datatype::clone(bool keepOriginalUniqueNodeId) {
  return new Datatype(this->getType());
}
std::string Datatype::getNodeName() const {
  return "Datatype";
}
void Datatype::accept(Visitor &v) {
  v.visit(*this);
}
