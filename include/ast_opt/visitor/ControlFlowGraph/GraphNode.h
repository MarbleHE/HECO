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
  READ = 0, WRITE = 1, READ_AND_WRITE = 2
};

typedef std::unordered_map<ScopedIdentifier, VariableAccessType> VarAccessMapType;

class GraphNode {
 private:
  /// A reference to the AST node that is associated to this GraphNode (i.e., statement).
  AbstractNode &astNode;

  /// A map of {VariableIdentifierScope : VariablesAccessType} pairs that shows which variables were accessed within
  /// this statement and the access type (read or write).
  VarAccessMapType variablesAccessMap;

  /// A reference to the edges of this GraphNode in the control flow graph.
  std::unique_ptr<NodeRelationship> controlFlowGraph;

  /// A reference to the edges of this GraphNode in the data flow graph.
  std::unique_ptr<NodeRelationship> dataFlowGraph;

 public:
  /// Creates a new GraphNode: a node in the control/data flow graph.
  /// \param originalNode The corresponding node in the AST classes. Must inherit from AbstractStatement as CFG/DFG only
  /// consists of statement nodes.
  explicit GraphNode(AbstractNode &originalNode);

  /// Creates a new GraphNode: a node in the control/data flow graph.
  /// This method is intended for TESTING PURPOSES ONLY and should not be used in real-world usage.
  /// \param originalNode The corresponding node in the AST classes. Must inherit from AbstractStatement as CFG/DFG only
  /// consists of statement nodes.
  /// \param relationshipType The relationship the given parentsToBeAdded nodes should be added to.
  /// \param parentsToBeAdded A list of (reference-wrapped) GraphNodes to be added as parent of the this node.
  GraphNode(AbstractNode &originalNode,
            RelationshipType relationshipType,
            const std::vector<std::reference_wrapper<GraphNode>> &parentsToBeAdded);

  /// Compares GraphNodes based on the unique node ID of the associated AST node, the number of children and parents
  /// in both the control flow graph and the data flow graph.
  /// \param t A reference-wrapped GraphNode object to compare this GraphNode with.
  /// \return True iff both GraphNodes are equal.
  bool operator==(const std::reference_wrapper<GraphNode> &t) const;

  /// Compares GraphNodes based on the unique node ID of the associated AST node, the number of children and parents
  /// in both the control flow graph and the data flow graph.
  /// \param t A GraphNode object to compare this GraphNode with.
  /// \return True iff both GraphNodes are equal.
  bool operator==(const GraphNode &t) const;

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
  [[nodiscard]] std::vector<ScopedIdentifier> getVariableAccessesByType(
      std::vector<VariableAccessType>);

  /// Set the variable accesses map by replacing the existing one.
  /// \param variablesAccesses The set of variable accesses to replace the current set with.
  void setAccessedVariables(VarAccessMapType &&variablesAccesses);

  /// Get all variables that are accessed within the GraphNode's associated AST statement.
  /// \return (A reference to)
  VarAccessMapType &getAccessedVariables();

  /// Get all variables that are accessed within the GraphNode's associated AST statement.
  /// \return (A const reference to)
  [[nodiscard]] const VarAccessMapType &getAccessedVariables() const;

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


//TODO: Specialize std::hash and std::<whatever comparator is> for GraphNode (or is it operator<?)

namespace std {
template<>
struct hash<GraphNode> {
  size_t operator()(const GraphNode &t) const {
    string hash = t.getAstNode().getUniqueNodeId();
    hash += +"|" + t.getControlFlowGraph().getChildren().size();
    hash += +"|" + t.getControlFlowGraph().getParents().size();
    hash += +"|" + t.getDataFlowGraph().getChildren().size();
    hash += +"|" + t.getDataFlowGraph().getParents().size();
    return std::hash<std::string>{}(hash);
  };
};

template<>
struct equal_to<GraphNode> {
  bool operator()(GraphNode const &s1, GraphNode const &s2) const {
    return s1 == s2;
  }
};

template<>
struct equal_to<std::reference_wrapper<GraphNode>> {
  bool operator()(std::reference_wrapper<GraphNode> const &s1, std::reference_wrapper<GraphNode> const &s2) const {
    return s1.get() == s2.get();
  }
};

template<>
struct hash<std::reference_wrapper<GraphNode>> {
  size_t operator()(const std::reference_wrapper<GraphNode> &t) const {
    return std::hash<GraphNode>{}(t);
  }
};

}
//class GraphNodeHashFunction {
//  /// This function must be passed to certain STL containers if they should contain GraphNodes, for example,
//  /// std::unordered_set<std::reference_wrapper<GraphNode>, GraphNodeHashFunction> mySet.
// public:
//  size_t operator()(const std::reference_wrapper<GraphNode> &t) const {
//    return std::hash<std::string>()(t.get().getAstNode().getUniqueNodeId())
//        ^ std::hash<size_t>()(t.get().getControlFlowGraph().getChildren().size())
//        ^ std::hash<size_t>()(t.get().getControlFlowGraph().getParents().size())
//        ^ std::hash<size_t>()(t.get().getDataFlowGraph().getChildren().size())
//        ^ std::hash<size_t>()(t.get().getDataFlowGraph().getParents().size());
//  }
//};
//
//struct GraphNodeComparator {
//  bool
//  operator()(const std::reference_wrapper<GraphNode> &obj1, const std::reference_wrapper<GraphNode> &obj2) const {
//    GraphNodeHashFunction hf;
//    return hf(obj1)==hf(obj2);
//  }
//};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_CONTROLFLOWGRAPH_GRAPHNODE_H_
