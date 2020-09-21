#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_CONTROLFLOWGRAPH_GRAPHNODE_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_CONTROLFLOWGRAPH_GRAPHNODE_H_

#include <set>

// include needed due to enum RelationshipType
#include "ast_opt/visitor/ControlFlowGraph/NodeRelationship.h"
#include "ast_opt/utilities/Scope.h"

class AbstractNode;
class NodeRelationship;

/// An enum class to describe how a variable was accessed. Supported values are READ and WRITE.
enum class VariableAccessType {
  READ = 0, WRITE = 1
};

typedef std::pair<ScopedIdentifier, VariableAccessType> VariableAccessPair;

class GraphNode {
 private:
  /// A reference to the AST node that is associated to this GraphNode (i.e., statement).
  AbstractNode &astNode;

  /// A map of {VariableIdentifierScope : VariablesAccessType} pairs that shows which variables were accessed within
  /// this statement and the access type (read or write).
  std::set<VariableAccessPair> variablesAccessMap;

  /// A reference to the edges of this GraphNode in the control flow graph.
  std::unique_ptr<NodeRelationship> controlFlowGraph;

  /// A reference to the edges of this GraphNode in the data flow graph.
  std::unique_ptr<NodeRelationship> dataFlowGraph;

 public:
  /// Creates a new GraphNode: a node in the control/data flow graph.
  /// \param originalNode The corresponding node in the AST classes. Must inherit from AbstractStatement as CFG/DFG only
  /// consists of statement nodes.
  explicit GraphNode(AbstractNode &originalNode);

//  /// Copy constructor.
//  /// \param other
//  GraphNode(const GraphNode &other);
//
//  /// Move constructor.
//  ///
//  /// \param other
//  GraphNode(GraphNode &&other) noexcept;
//
//  /// Copy assignment constructor.
//  /// \param other
//  /// \return
//  GraphNode &operator=(const GraphNode &other);
//
//  /// Move assignment constructor.
//  /// \param other
//  /// \return
//  GraphNode &operator=(GraphNode &&other) noexcept;

  /// Retrieves the relationship (i.e., information about edges from this node) that corresponds to the given
  /// relationship type.
  /// \param relationshipType The type of relationship that should be retrieved.
  /// \return (A reference to) the relationship object corresponding to the given type.
  NodeRelationship &getRelationship(RelationshipType relationshipType);

  /// Retrieves the relationship (i.e., information about edges from this node) that corresponds to the given
  /// relationship type.
  /// \param relationshipType The type of relationship that should be retrieved.
  /// \return (A const reference to) the relationship object corresponding to the given type.
  [[nodiscard]] const NodeRelationship &getRelationship(RelationshipType relationshipType) const;

  /// Returns the set of all VarAccessPairs that match the given VariableAccessType.
  /// \param accessType The access type acting as filter for the recorded variable accesses.
  /// \return A set containing (copies from) variable accesses of this GraphNode.
  [[nodiscard]] std::set<VariableAccessPair> getVariableAccessesByType(
      VariableAccessType accessType);

  /// Set the variable accesses map by replacing the existing one.
  /// \param variablesAccesses The set of variable accesses to replace the current set with.
  void setAccessedVariables(std::set<VariableAccessPair> &&variablesAccesses);

  /// Get all variables that are accessed within the GraphNode's associated AST statement.
  /// \return (A reference to)
  std::set<VariableAccessPair> &getAccessedVariables();

  /// Get all variables that are accessed within the GraphNode's associated AST statement.
  /// \return (A const reference to)
  [[nodiscard]] const std::set<VariableAccessPair> &getAccessedVariables() const;

  /// Get the AST node associated with this GraphNode.
  /// \return (A reference to) the associated AST object.
  AbstractNode &getAstNode();

  /// Get the AST node associated with this GraphNode.
  /// \return (A const reference to) the associated AST object.
  [[nodiscard]] const AbstractNode &getAstNode() const;

  /// Get the edges of the control flow graph.
  /// \return (A reference to) the NodeRelationship of the control flow graph.
  NodeRelationship &getControlFlowGraph();

  /// Get the edges of the control flow graph.
  /// \return (A const reference to) the NodeRelationship of the control flow graph.
  [[nodiscard]] const NodeRelationship &getControlFlowGraph() const;

  /// Get the edges in the data flow graph.
  /// \return (A reference to) the NodeRelationship of the dataflow graph.
  NodeRelationship &getDataFlowGraph();

  /// Get the edges in the data flow graph.
  /// \return (A const reference to) the NodeRelationship of the dataflow graph.
  [[nodiscard]] const NodeRelationship &getDataFlowGraph() const;
};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_CONTROLFLOWGRAPH_GRAPHNODE_H_
