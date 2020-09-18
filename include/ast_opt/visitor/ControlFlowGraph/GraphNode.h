#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_CONTROLFLOWGRAPH_GRAPHNODE_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_CONTROLFLOWGRAPH_GRAPHNODE_H_

#include <set>

#include "ast_opt/ast/AbstractNode.h"
#include "ast_opt/visitor/ControlFlowGraph/NodeRelationship.h"
#include "ast_opt/visitor/ControlFlowGraph/ControlFlowGraphVisitor.h"

/// An enum class to describe how a variable was accessed. Supported values are READ and WRITE.
enum class VariableAccessType { READ = 0, WRITE = 1 };

// TODO: Implement scope logic and replace this.
struct Scope {};

typedef std::tuple<Scope, std::string> VariableIdentifierScope;
typedef std::pair<VariableIdentifierScope, VariableAccessType> VarAccessPair;

class GraphNode {
 private:

  ///
  AbstractNode &astNode;

  ///
  std::unordered_set<VarAccessPair> variablesAccessMap;

  ///
  std::unique_ptr<NodeRelationship> controlFlowGraph;

  ///
  std::unique_ptr<NodeRelationship> dataFlowGraph;

 public:
  ///
  /// \param originalNode
  explicit GraphNode(AbstractNode &originalNode);

  ///
  /// \param relationshipType
  /// \return
  NodeRelationship &getRelationship(RelationshipType relationshipType);

  ///
  /// \param relationshipType
  /// \return
  [[nodiscard]] const NodeRelationship &getRelationship(RelationshipType relationshipType) const;

  ///
  /// \param accessType
  /// \return
  [[nodiscard]] std::unordered_set<VarAccessPair> getVariableAccessesByType(
      VariableAccessType accessType);

  ///
  /// \param variablesAccesses
  void setAccessedVariables(std::unordered_set<VarAccessPair> &&variablesAccesses);

  ///
  /// \return
  std::unordered_set<VarAccessPair> &getAccessedVariables();

  ///
  /// \return
  [[nodiscard]] const std::unordered_set<VarAccessPair> &getAccessedVariables() const;

  ///
  /// \return
  AbstractNode &getAstNode();

  ///
  /// \return
  [[nodiscard]] const AbstractNode &getAstNode() const;

  ///
  /// \return
  NodeRelationship &getControlFlowGraph();

  ///
  /// \return
  [[nodiscard]] const NodeRelationship &getControlFlowGraph() const;

  ///
  /// \return
  NodeRelationship &getDataFlowGraph();

  ///
  /// \return
  [[nodiscard]] const NodeRelationship &getDataFlowGraph() const;
};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_CONTROLFLOWGRAPH_GRAPHNODE_H_
