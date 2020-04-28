#include "ast_opt/ast/ParameterList.h"
#include "ast_opt/ast/FunctionParameter.h"

std::string ParameterList::getNodeType() const {
  return "ParameterList";
}

void ParameterList::accept(Visitor &v) {
  v.visit(*this);
}

ParameterList *ParameterList::clone(bool keepOriginalUniqueNodeId) {
  auto childrenCopy = children;
  for (auto &c: childrenCopy) {
    c = c->clone(keepOriginalUniqueNodeId);
  }
  auto clonedNode = new ParameterList();
  clonedNode->addChildren(childrenCopy);
  clonedNode->updateClone(keepOriginalUniqueNodeId, this);
  return clonedNode;
}

ParameterList::ParameterList(std::vector<FunctionParameter *> parameters) {
  for (auto &fp : parameters) {
    addChild(fp);
  }
}

std::vector<FunctionParameter *> ParameterList::getParameters() {
  std::vector<FunctionParameter *> params;
  params.reserve(children.size());
  for (auto &n : children) {
    params.emplace_back(dynamic_cast<FunctionParameter *>(n));
  }
  return params;
}

int ParameterList::getMaxNumberChildren() {
  return -1;
}

bool ParameterList::supportsCircuitMode() {
  return true;
}

std::string ParameterList::toString(bool printChildren) const {
  return AbstractNode::generateOutputString(printChildren, {});
}
