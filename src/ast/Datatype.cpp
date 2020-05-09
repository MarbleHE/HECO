#include "ast_opt/ast/Datatype.h"
#include "ast_opt/ast/LiteralFloat.h"
#include "ast_opt/ast/LiteralInt.h"
#include "ast_opt/ast/LiteralBool.h"
#include "ast_opt/ast/LiteralString.h"

Datatype::Datatype(const std::string &type) {
  static const std::unordered_map<std::string, Types> string_to_types = {
      {"int", Types::INT},
      {"float", Types::FLOAT},
      {"string", Types::STRING},
      {"bool", Types::BOOL}};
  auto result = string_to_types.find(type);
  if (result==string_to_types.end()) {
    throw std::invalid_argument(
        "Unsupported datatype given: " + type + ". See the supported datatypes in Datatype.h.");
  }
  val = result->second;
}

json Datatype::toJson() const {
  json j = {{"type", getNodeType()},
            {"encrypted", isEncrypted()},
            {"specifier", enumToString(val)}
  };
  return j;
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

std::string Datatype::toString(bool printChildren) const {
  return AbstractNode::generateOutputString(printChildren,
                                            {(isEncrypted() ? "encrypted" : "plaintext"), enumToString(val)});
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

void Datatype::setEncrypted(bool isEncrypted) {
  Datatype::encrypted = isEncrypted;
}

AbstractNode *Datatype::cloneFlat() {
  auto datatype = new Datatype(this->getType(), isEncrypted());
  datatype->setUniqueNodeId(this->getUniqueNodeId());
  return datatype;
}

AbstractNode *Datatype::clone(bool keepOriginalUniqueNodeId) {
  auto clonedNode = new Datatype(this->getType(), isEncrypted());
  clonedNode->updateClone(keepOriginalUniqueNodeId, this);
  return clonedNode;
}

std::string Datatype::getNodeType() const {
  return "Datatype";
}

void Datatype::accept(Visitor &v) {
  v.visit(*this);
}

bool Datatype::supportsCircuitMode() {
  return true;
}
