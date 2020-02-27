#include "VarDecl.h"
#include <utility>
#include "LiteralInt.h"
#include "LiteralBool.h"
#include "LiteralFloat.h"
#include "ArithmeticExpr.h"
#include "Ast.h"

json VarDecl::toJson() const {
  json j = {{"type", getNodeName()},
            {"identifier", identifier},
            {"datatype", getDatatype() ? getDatatype()->toString() : ""}};
  if (getInitializer()!=nullptr) {
    j["initializer"] = getInitializer()->toJson();
  }
  return j;
}

VarDecl::VarDecl(const std::string &, void *) {
  throw std::invalid_argument("VarDecl(std::string, AbstractExpr*) not accepted as datatype cannot be determined. "
                              "Use VarDecl(std::string, Types, AbstractExpr*) or one of the other constructors.");
}

VarDecl::VarDecl(std::string name, Datatype *datatype, AbstractExpr *initializer) {
  setAttributes(std::move(name), datatype, initializer);
}

VarDecl::VarDecl(std::string name, Types datatype, AbstractExpr *initializer) {
  setAttributes(std::move(name), new Datatype(datatype), initializer);
}

VarDecl::VarDecl(std::string name, std::string valueAssignedTo) {
  setAttributes(std::move(name),
                new Datatype(Types::STRING),
                new LiteralString(std::move(valueAssignedTo)));
}

VarDecl::VarDecl(std::string name, int valueAssignedTo) {
  setAttributes(std::move(name), new Datatype(Types::INT), new LiteralInt(valueAssignedTo));
}

VarDecl::VarDecl(std::string name, Datatype *datatype) {
  setAttributes(std::move(name), new Datatype(Types::INT), nullptr);
}

VarDecl::VarDecl(std::string name, float valueAssignedTo) {
  setAttributes(std::move(name), new Datatype(Types::FLOAT), new LiteralFloat(valueAssignedTo));
}

VarDecl::VarDecl(std::string name, bool valueAssignedTo) {
  setAttributes(std::move(name), new Datatype(Types::BOOL), new LiteralBool(valueAssignedTo));
}

VarDecl::VarDecl(std::string name, const char *valueAssignedTo)
    : VarDecl(std::move(name), std::string(valueAssignedTo)) {}

void VarDecl::setAttributes(std::string varIdentifier, Datatype *varDatatype, AbstractExpr *varValue) {
  // handle primitive attributes
  this->identifier = std::move(varIdentifier);
  // handle attributes that are itself nodes
  removeChildren();
  addChildren({varDatatype, varValue}, true);
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
  return getChildAtIndex(0, true)->castTo<Datatype>();
}

AbstractExpr *VarDecl::getInitializer() const {
  auto initializer = getChildAtIndex(1, true);
  if (initializer==nullptr)
    return nullptr;
  return initializer->castTo<AbstractExpr>();
}

ArithmeticExpr *VarDecl::contains(ArithmeticExpr *aexpTemplate, ArithmeticExpr *excludedSubtree) {
  return this->getInitializer()->contains(aexpTemplate, excludedSubtree);
}

VarDecl::~VarDecl() {
  for (auto &c : getChildren()) delete c;
}

std::string VarDecl::getVarTargetIdentifier() {
  return this->getIdentifier();
}

bool VarDecl::isEqual(AbstractStatement *as) {
  if (auto otherVarDecl = dynamic_cast<VarDecl *>(as)) {
    return (this->getIdentifier()==otherVarDecl->getIdentifier())
        && (*this->getDatatype()==*otherVarDecl->getDatatype())
        && (this->getInitializer()->isEqual(otherVarDecl->getInitializer()));
  }
  return false;
}

bool VarDecl::supportsCircuitMode() {
  return true;
}

int VarDecl::getMaxNumberChildren() {
  return 2;
}

VarDecl *VarDecl::clone(bool keepOriginalUniqueNodeId) {
  auto clonedNode = new VarDecl(this->getVarTargetIdentifier(),
                                this->getDatatype()->getType(),
                                getInitializer()->clone(keepOriginalUniqueNodeId)->castTo<AbstractExpr>());
  if (keepOriginalUniqueNodeId) clonedNode->setUniqueNodeId(this->getUniqueNodeId());
  if (this->isReversed) clonedNode->swapChildrenParents();
  return clonedNode;
}

std::string VarDecl::toString() const {
  return identifier;
}
