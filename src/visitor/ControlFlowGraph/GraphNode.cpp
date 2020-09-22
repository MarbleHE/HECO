#include "ast_opt/visitor/ControlFlowGraph/GraphNode.h"
#include "ast_opt/ast/AbstractNode.h"
#include "ast_opt/visitor/ControlFlowGraph/NodeRelationship.h"

GraphNode::GraphNode(AbstractNode &originalNode) : astNode(originalNode) {
  controlFlowGraph = std::make_unique<NodeRelationship>(RelationshipType::CTRL_FLOW_GRAPH, *this);
  dataFlowGraph = std::make_unique<NodeRelationship>(RelationshipType::DATA_FLOW_GRAPH, *this);
}

GraphNode::GraphNode(AbstractNode &originalNode,
                     RelationshipType relationshipType,
                     const std::vector<std::reference_wrapper<GraphNode>> &parentsToBeAdded)
    : astNode(originalNode) {
  controlFlowGraph = std::make_unique<NodeRelationship>(RelationshipType::CTRL_FLOW_GRAPH, *this);
  dataFlowGraph = std::make_unique<NodeRelationship>(RelationshipType::DATA_FLOW_GRAPH, *this);
  for (auto &parentNode : parentsToBeAdded) {
    getRelationship(relationshipType).addParent(parentNode);
  }
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

const std::set<VariableAccessPair> &GraphNode::getAccessedVariables() const {
  return variablesAccessMap;
}

std::set<VariableAccessPair> &GraphNode::getAccessedVariables() {
  return variablesAccessMap;
}

std::set<VariableAccessPair> GraphNode::getVariableAccessesByType(VariableAccessType accessType) {
  std::set<VariableAccessPair> resultSet;
  auto hasRequestedAccessType = [&accessType](const VariableAccessPair &p) { return p.second==accessType; };
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

void GraphNode::setAccessedVariables(std::set<VariableAccessPair> &&variablesAccesses) {
  this->variablesAccessMap = std::move(variablesAccesses);
}

bool GraphNode::operator==(const std::reference_wrapper<GraphNode> &t) const {
  return
    // same uniqueNodeId
      this->getAstNode().getUniqueNodeId()
          ==t.get().getAstNode().getUniqueNodeId()
          // same number of control flow graph children
          && this->getControlFlowGraph().getChildren().size()
              ==t.get().getControlFlowGraph().getChildren().size()
              // same number of control flow graph parents
          && this->getControlFlowGraph().getParents().size()
              ==t.get().getControlFlowGraph().getParents().size()
              // same number of data flow graph children
          && this->getDataFlowGraph().getChildren().size()
              ==t.get().getDataFlowGraph().getChildren().size()
              // same number of data flow graph parents
          && this->getDataFlowGraph().getParents().size()
              ==t.get().getDataFlowGraph().getParents().size();
}

