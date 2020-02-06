#include "../../include/ast/Group.h"
#include "BinaryExpr.h"

json Group::toJson() const {
  json j;
  j["type"] = getNodeName();
  j["expr"] = getExpr() ? getExpr()->toJson() : "";
  return j;
}

void Group::accept(Visitor &v) {
  v.visit(*this);
}

std::string Group::getNodeName() const {
  return "Group";
}

Group::Group(AbstractExpr* expr) {
  setAttributes(expr);
}

AbstractExpr* Group::getExpr() const {
  return reinterpret_cast<AbstractExpr*>(getChildAtIndex(0));
}

Group::~Group() {
  for (auto &c : getChildren()) delete c;
}

BinaryExpr* Group::contains(BinaryExpr* bexpTemplate, AbstractExpr* excludedSubtree) {
  if (auto* child = dynamic_cast<BinaryExpr*>(getExpr())) {
    return child->contains(bexpTemplate, excludedSubtree);
  }
  return nullptr;
}

Literal* Group::evaluate(Ast &ast) {
  return this->getExpr()->evaluate(ast);
}

bool Group::supportsCircuitMode() {
  return true;
}

int Group::getMaxNumberChildren() {
  return 1;
}

void Group::setAttributes(AbstractExpr* expression) {
  removeChildren();
  addChildren({expression}, false);
  Node::addParentTo(this, {expression});
}

