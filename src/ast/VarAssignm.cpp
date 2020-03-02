#include <utility>
#include "VarAssignm.h"
#include "ArithmeticExpr.h"

json VarAssignm::toJson() const {
  json j;
  j["type"] = getNodeType();
  j["identifier"] = this->identifier;
  j["value"] = getValue() ? getValue()->toJson() : "";
  return j;
}

VarAssignm::VarAssignm(std::string identifier, AbstractExpr *value) : identifier(std::move(identifier)) {
  setAttribute(value);
}

void VarAssignm::accept(Visitor &v) {
  v.visit(*this);
}

const std::string &VarAssignm::getIdentifier() const {
  return identifier;
}

AbstractExpr *VarAssignm::getValue() const {
  return reinterpret_cast<AbstractExpr *>(getChildAtIndex(0, true));
}

std::string VarAssignm::getNodeType() const {
  return "VarAssignm";
}

AbstractBinaryExpr *VarAssignm::contains(AbstractBinaryExpr *aexpTemplate, ArithmeticExpr *excludedSubtree) {
  return this->getValue()->contains(aexpTemplate, excludedSubtree);
}

VarAssignm::~VarAssignm() {
  delete getChildAtIndex(0);
}

std::string VarAssignm::getVarTargetIdentifier() {
  return this->getIdentifier();
}

bool VarAssignm::isEqual(AbstractStatement *as) {
  if (auto otherVarAssignm = dynamic_cast<VarAssignm *>(as)) {
    return this->getIdentifier()==otherVarAssignm->getIdentifier() &&
        this->getValue()->isEqual(otherVarAssignm->getValue());
  }
  return false;
}

bool VarAssignm::supportsCircuitMode() {
  return true;
}

int VarAssignm::getMaxNumberChildren() {
  return 1;
}

void VarAssignm::setAttribute(AbstractExpr *assignmentValue) {
  removeChildren();
  addChildren({assignmentValue}, true);
}

VarAssignm *VarAssignm::clone(bool keepOriginalUniqueNodeId) {
  auto clonedNode = new VarAssignm(this->getIdentifier(),
                                   this->getValue()->clone(keepOriginalUniqueNodeId)->castTo<AbstractExpr>());
  if (keepOriginalUniqueNodeId) clonedNode->setUniqueNodeId(this->getUniqueNodeId());
  if (this->isReversed) clonedNode->swapChildrenParents();
  return clonedNode;
}

std::string VarAssignm::toString(bool printChildren) const {
  return AbstractNode::generateOutputString(printChildren, {this->identifier});
}
