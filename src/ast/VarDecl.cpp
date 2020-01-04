#include "../../include/ast/VarDecl.h"

#include <utility>
#include <LiteralInt.h>
#include "BinaryExpr.h"
#include "Group.h"
#include "Ast.h"

json VarDecl::toJson() const {
  json j = {{"type", getNodeName()},
            {"identifier", identifier},
            {"datatype", datatype}};
  if (this->initializer != nullptr) {
    j["initializer"] = this->initializer->toJson();
  }
  return j;
}

VarDecl::VarDecl(std::string name, std::string datatype, AbstractExpr* initializer)
    : identifier(std::move(std::move(name))), datatype(std::move(std::move(datatype))), initializer(initializer) {}

VarDecl::VarDecl(std::string name, std::string datatype) : identifier(std::move(std::move(name))),
                                                           datatype(std::move(std::move(datatype))) {
  this->initializer = nullptr;
}

VarDecl::VarDecl(std::string name, const std::string &datatype, int i)
    : identifier(std::move(std::move(name))), datatype(datatype) {
  if (datatype == "int") {
    initializer = new LiteralInt(i);
  } else {
    std::string errorMsg("Datatype 'int' and provided value (" + std::to_string(i) + ") do not match!");
    throw std::logic_error(errorMsg);
  }
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

const std::string &VarDecl::getDatatype() const {
  return datatype;
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
        && (this->getDatatype() == otherVarDecl->getDatatype())
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
