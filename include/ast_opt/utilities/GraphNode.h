#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_UTILITIES_GRAPHNODE_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_UTILITIES_GRAPHNODE_H_

#include <set>
#include <string>
#include <utility>
#include "ast_opt/ast/AbstractNode.h"
#include "ast_opt/utilities/NodeRelationship.h"
#include "ast_opt/visitor/ControlFlowGraphVisitor.h"

class GraphNode {
 private:
  AbstractNode *refToOriginalNode{nullptr};

  std::set<std::pair<std::string, AccessType>> accessedVariables;

  NodeRelationship *controlFlowGraph = new NodeRelationship{RelationshipType::CTRL_FLOW_GRAPH, this};

  NodeRelationship *dataFlowGraph = new NodeRelationship{RelationshipType::DATA_FLOW_GRAPH, this};

 public:
  GraphNode() = default;

  GraphNode(RelationshipType relType, std::initializer_list<GraphNode *> parentsToBeAdded);

  explicit GraphNode(AbstractNode *originalNode);

  [[nodiscard]] NodeRelationship *getControlFlowGraph() const;

  [[nodiscard]] NodeRelationship *getDataFlowGraph() const;

  [[nodiscard]] NodeRelationship *getRelationship(RelationshipType rel) const;

  [[nodiscard]] const std::set<std::pair<std::string, AccessType>> &getAccessedVariables() const;

  [[nodiscard]] AbstractNode *getRefToOriginalNode() const;

  void setAccessedVariables(std::set<std::pair<std::string, AccessType>> set);

  [[nodiscard]] std::set<std::string> getVariables(AccessType accessTypeFilter) const;
};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_UTILITIES_GRAPHNODE_H_
