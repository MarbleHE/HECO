#include "ast_opt/utilities/GraphNode.h"

GraphNode::GraphNode(RelationshipType relType, std::initializer_list<GraphNode *> parentsToBeAdded) {
  for (auto &c : parentsToBeAdded) {
    getRelationship(relType)->addParent(c);
  }
}

GraphNode::GraphNode(AbstractNode *originalNode) {
  refToOriginalNode = originalNode;
}

NodeRelationship *GraphNode::getControlFlowGraph() const {
  return controlFlowGraph;
}

NodeRelationship *GraphNode::getDataFlowGraph() const {
  return dataFlowGraph;
}

const std::set<std::pair<ScopedVariable, AccessType>> & GraphNode::getAccessedVariables() const {
  return accessedVariables;
}

std::set<ScopedVariable> GraphNode::getVariables(AccessType accessTypeFilter) const {
  std::set<ScopedVariable> filteredVariables;
  for (auto &[var, acType] : accessedVariables) {
    if (acType==accessTypeFilter) filteredVariables.insert(var);
  }
  return filteredVariables;
}

AbstractNode *GraphNode::getRefToOriginalNode() const {
  return refToOriginalNode;
}

NodeRelationship *GraphNode::getRelationship(RelationshipType rel) const {
  if (rel==RelationshipType::CTRL_FLOW_GRAPH) {
    return getControlFlowGraph();
  } else if (rel==RelationshipType::DATA_FLOW_GRAPH) {
    return getDataFlowGraph();
  } else {
    throw std::invalid_argument("Unknown RelationshipType!");
  }
}

void GraphNode::setAccessedVariables(std::set<std::pair<ScopedVariable, AccessType>> set) {
  accessedVariables = std::move(set);
}

