#include "GraphNode.h"

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

const std::set<std::pair<std::string, AccessType>> &GraphNode::getAccessedVariables() const {
  return accessedVariables;
}

std::set<std::pair<std::string, AccessType>> GraphNode::getVariables(AccessType accessType) const {
  std::set<std::pair<std::string, AccessType>> writtenVars;
  std::copy_if(accessedVariables.begin(), accessedVariables.end(), std::inserter(writtenVars, writtenVars.begin()),
               [accessType](const std::pair<std::string, AccessType> &p) {
                 return p.second==accessType;
               });
  return writtenVars;
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

void GraphNode::setAccessedVariables(std::set<std::pair<std::string, AccessType>> set) {
  accessedVariables = std::move(set);
}

