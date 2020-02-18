#include <utility>
#include "VarAssignm.h"
#include "BinaryExpr.h"

json VarAssignm::toJson() const {
  json j;
  j["type"] = getNodeName();
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

std::string VarAssignm::getNodeName() const {
  return "VarAssignm";
}

BinaryExpr *VarAssignm::contains(BinaryExpr *bexpTemplate, BinaryExpr *excludedSubtree) {
  return this->getValue()->contains(bexpTemplate, excludedSubtree);
}

VarAssignm::~VarAssignm() {
  delete getChildAtIndex(0);
}

std::string VarAssignm::getVarTargetIdentifier() {
  return this->getIdentifier();
}

bool VarAssignm::isEqual(AbstractStatement *as) {
  if (auto otherVarAssignm = dynamic_cast<VarAssignm *>(as)) {
    return this->getIdentifier() == otherVarAssignm->getIdentifier() &&
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
  addChildren({assignmentValue}, false);
  addParentTo(this, {assignmentValue});
}

VarAssignm *VarAssignm::clone(bool keepOriginalUniqueNodeId) {
  auto clonedNode =  new VarAssignm(this->getIdentifier(),
                        this->getValue()->clone(keepOriginalUniqueNodeId)->castTo<AbstractExpr>());
  if (keepOriginalUniqueNodeId) clonedNode->setUniqueNodeId(this->getUniqueNodeId());
  if (this->isReversed) clonedNode->swapChildrenParents();
  return clonedNode;
}
