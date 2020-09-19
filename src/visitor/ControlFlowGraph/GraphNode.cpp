#include "ast_opt/visitor/ControlFlowGraph/GraphNode.h"
#include "ast_opt/visitor/ControlFlowGraph/NodeRelationship.h"

GraphNode::GraphNode(AbstractNode &originalNode) : astNode(originalNode) {
  controlFlowGraph = std::make_unique<NodeRelationship>(RelationshipType::CTRL_FLOW_GRAPH, *this);
  dataFlowGraph = std::make_unique<NodeRelationship>(RelationshipType::DATA_FLOW_GRAPH, *this);
}

NodeRelationship &GraphNode::getControlFlowGraph() {
  return *controlFlowGraph;
}

const NodeRelationship &GraphNode::getControlFlowGraph() const {
  return *controlFlowGraph;
}

NodeRelationship &GraphNode::getDataFlowGraph() {
  return *dataFlowGraph;
}

const NodeRelationship &GraphNode::getDataFlowGraph() const {
  return *dataFlowGraph;
}

const std::set<VarAccessPair> &GraphNode::getAccessedVariables() const {
  return variablesAccessMap;
}

std::set<VarAccessPair> &GraphNode::getAccessedVariables() {
  return variablesAccessMap;
}

std::set<VarAccessPair> GraphNode::getVariableAccessesByType(VariableAccessType accessType) {
  std::set<VarAccessPair> resultSet;
  auto hasRequestedAccessType = [&accessType](const VarAccessPair &p) { return p.second==accessType; };
  std::copy_if(variablesAccessMap.begin(),
               variablesAccessMap.end(),
               std::inserter(resultSet, resultSet.begin()),
               hasRequestedAccessType);
  return resultSet;
}

AbstractNode &GraphNode::getAstNode() {
  return astNode;
}

const AbstractNode &GraphNode::getAstNode() const {
  return astNode;
}

NodeRelationship &GraphNode::getRelationship(RelationshipType relationshipType) {
  // removes const from result of const counterpart, see https://stackoverflow.com/a/856839/3017719
  return const_cast<NodeRelationship &>(const_cast<const GraphNode *>(this)->getRelationship(relationshipType));
}

const NodeRelationship &GraphNode::getRelationship(RelationshipType relationshipType) const {
  if (relationshipType==RelationshipType::CTRL_FLOW_GRAPH) {
    return getControlFlowGraph();
  } else if (relationshipType==RelationshipType::DATA_FLOW_GRAPH) {
    return getDataFlowGraph();
  } else {
    throw std::invalid_argument("Unknown RelationshipType given!");
  }
}

void GraphNode::setAccessedVariables(std::set<VarAccessPair> &&variablesAccesses) {
  this->variablesAccessMap = std::move(variablesAccesses);
}

