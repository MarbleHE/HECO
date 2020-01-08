#include "../../include/ast/VarDecl.h"

#include <utility>
#include "LiteralInt.h"
#include "LiteralBool.h"
#include "LiteralFloat.h"
#include "BinaryExpr.h"
#include "Group.h"
#include "Ast.h"
#include "../../include/utilities/Datatypes.h"

json VarDecl::toJson() const {
  json j = {{"type", getNodeName()},
            {"identifier", identifier},
            {"datatype", datatype->toString()}};
  if (this->initializer != nullptr) {
    j["initializer"] = this->initializer->toJson();
  }
  return j;
}

VarDecl::VarDecl(std::string name, const std::string &datatype, AbstractExpr* initializer)
    : identifier(std::move(std::move(name))), initializer(initializer) {
  this->datatype = new Datatype(datatype);
}

VarDecl::VarDecl(std::string name, std::string value) : identifier(std::move(std::move(name))) {
  this->initializer = nullptr;
  this->datatype = new Datatype(TYPES::STRING);
  this->initializer = new LiteralString(std::move(value));
}

VarDecl::VarDecl(std::string name, int value) : identifier(std::move(std::move(name))) {
  this->datatype = new Datatype(TYPES::INT);
  this->initializer = new LiteralInt(value);
}

VarDecl::VarDecl(std::string name, float value) : identifier(std::move(std::move(name))) {
  this->datatype = new Datatype(TYPES::INT);
  this->initializer = new LiteralFloat(value);
}

VarDecl::VarDecl(std::string name, bool value) : identifier(std::move(std::move(name))) {
  this->datatype = new Datatype(TYPES::INT);
  this->initializer = new LiteralBool(value);
}

void VarDecl::accept(Visitor &v) {
  v.visit(*this);
}

std::string VarDecl::getNodeName() const {
  return "VarDecl";
}

const std::string &VarDecl::getIdentifier() const {
  return identifier;
}

AbstractExpr* VarDecl::getInitializer() const {
  return initializer;
}

BinaryExpr* VarDecl::contains(BinaryExpr* bexpTemplate, BinaryExpr* excludedSubtree) {
  return this->getInitializer()->contains(bexpTemplate, excludedSubtree);
}

VarDecl::~VarDecl() {
  delete initializer;
}

std::string VarDecl::getVarTargetIdentifier() {
  return this->getIdentifier();
}

bool VarDecl::isEqual(AbstractStatement* as) {
  if (auto otherVarDecl = dynamic_cast<VarDecl*>(as)) {
    return (this->getIdentifier() == otherVarDecl->getIdentifier())
        && (*this->getDatatype() == *otherVarDecl->getDatatype())
        && (this->getInitializer()->isEqual(otherVarDecl->getInitializer()));
  }
  return false;
}

Literal* VarDecl::evaluate(Ast &ast) {
  if (this->getInitializer() != nullptr) {
    auto value = this->getInitializer()->evaluate(ast);
    ast.updateVarValue(this->getIdentifier(), value);
    return value;
  }
  return nullptr;
}

Datatype* VarDecl::getDatatype() const {
  return datatype;
}
