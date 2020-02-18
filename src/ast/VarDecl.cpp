#include "VarDecl.h"
#include <utility>
#include "LiteralInt.h"
#include "LiteralBool.h"
#include "LiteralFloat.h"
#include "BinaryExpr.h"
#include "Ast.h"

json VarDecl::toJson() const {
  json j = {{"type",       getNodeName()},
            {"identifier", identifier},
            {"datatype",   getDatatype() ? getDatatype()->toString() : ""}};
  if (getInitializer() != nullptr) {
    j["initializer"] = getInitializer()->toJson();
  }
  return j;
}

VarDecl::VarDecl(std::string, void *) {
  throw std::invalid_argument("VarDecl(std::string, AbstractExpr*) not accepted as datatype cannot be determined. "
                              "Use VarDecl(std::string, TYPES, AbstractExpr*) or one of the other constructors.");
}

VarDecl::VarDecl(std::string name, TYPES datatype, AbstractExpr *initializer) {
  setAttributes(std::move(name), new Datatype(datatype), initializer);
}

VarDecl::VarDecl(std::string name, std::string valueAssignedTo) {
  setAttributes(std::move(name),
                new Datatype(TYPES::STRING),
                new LiteralString(std::move(valueAssignedTo)));
}

VarDecl::VarDecl(std::string name, int valueAssignedTo) {
  setAttributes(std::move(name), new Datatype(TYPES::INT), new LiteralInt(valueAssignedTo));
}

VarDecl::VarDecl(std::string name, float valueAssignedTo) {
  setAttributes(std::move(name), new Datatype(TYPES::FLOAT), new LiteralFloat(valueAssignedTo));
}

VarDecl::VarDecl(std::string name, bool valueAssignedTo) {
  setAttributes(std::move(name), new Datatype(TYPES::BOOL), new LiteralBool(valueAssignedTo));
}

VarDecl::VarDecl(std::string name, const char *valueAssignedTo)
    : VarDecl(std::move(name), std::string(valueAssignedTo)) {}

void VarDecl::setAttributes(std::string varIdentifier, Datatype *varDatatype, AbstractExpr *varValue) {
  // handle primitive attributes
  this->identifier = std::move(varIdentifier);
  // handle attributes that are itself nodes
  removeChildren();
  addChildren({varDatatype, varValue}, false);
  AbstractNode::addParentTo(this, {varDatatype, varValue});
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

Datatype *VarDecl::getDatatype() const {
  return reinterpret_cast<Datatype *>(getChildAtIndex(0, true));
}

AbstractExpr *VarDecl::getInitializer() const {
  return reinterpret_cast<AbstractExpr *>(getChildAtIndex(1, true));
}

BinaryExpr *VarDecl::contains(BinaryExpr *bexpTemplate, BinaryExpr *excludedSubtree) {
  return this->getInitializer()->contains(bexpTemplate, excludedSubtree);
}

VarDecl::~VarDecl() {
  for (auto &c : getChildren()) delete c;
}

std::string VarDecl::getVarTargetIdentifier() {
  return this->getIdentifier();
}

bool VarDecl::isEqual(AbstractStatement *as) {
  if (auto otherVarDecl = dynamic_cast<VarDecl *>(as)) {
    return (this->getIdentifier() == otherVarDecl->getIdentifier())
           && (*this->getDatatype() == *otherVarDecl->getDatatype())
           && (this->getInitializer()->isEqual(otherVarDecl->getInitializer()));
  }
  return false;
}

std::vector<Literal *> VarDecl::evaluate(Ast &ast) {
  if (this->getInitializer() != nullptr) {
    auto value = this->getInitializer()->evaluate(ast).front();
    ast.updateVarValue(this->getIdentifier(), value);
    return std::vector<Literal *>({value});
  } else {
    ast.updateVarValue(this->getIdentifier(), nullptr);
    return std::vector<Literal *>();
  }
}

bool VarDecl::supportsCircuitMode() {
  return true;
}

int VarDecl::getMaxNumberChildren() {
  return 2;
}

AbstractNode *VarDecl::createClonedNode(bool keepOriginalUniqueNodeId) {
  return new VarDecl(this->getVarTargetIdentifier(),
                     this->getDatatype()->getType(),
                     getInitializer()->cloneRecursiveDeep(keepOriginalUniqueNodeId)->castTo<AbstractExpr>());
}
