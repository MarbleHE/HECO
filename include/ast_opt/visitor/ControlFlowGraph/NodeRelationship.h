#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_CONTROLFLOWGRAPH_NODERELATIONSHIP_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_CONTROLFLOWGRAPH_NODERELATIONSHIP_H_

#include <unordered_set>
#include <vector>
#include <iostream>
#include <set>
#include <stack>
#include <sstream>

class GraphNode;

enum class RelationshipType { CTRL_FLOW_GRAPH, DATA_FLOW_GRAPH };

// forward declarations
class GraphNode;

class NodeRelationship {
 private:
  /// The relationship type of this node relationship, i.e., whether this is an edge in the control flow graph or in
  /// the data flow graph as the same GraphNode is used in both.
  RelationshipType relationshipType;

  /// A reference to the GraphNode objects this edge belongs to.
  GraphNode &graphNode;

  /// A vector of children nodes, i.e., the nodes that this edge (NodeRelationship) points to (outgoing edges).
  std::vector<std::reference_wrapper<GraphNode>> children;

  /// A vector of parent nodes, i.e., the nodes that point to this node (incoming edges).
  std::vector<std::reference_wrapper<GraphNode>> parents;

 public:
  /// Creates a new NodeRelationship object: An edge in the control flow or data flow graph.
  /// \param relationshipType Indicates whether this edge belongs to the control flow or data flow graph.
  /// \param graphNode A reference to the graph this edge is associated with.
  NodeRelationship(RelationshipType relationshipType, GraphNode &graphNode);

  ///
  /// \param child
  /// \param addBackreference
  void addChild(GraphNode &child, bool addBackreference = true);

  ///
  /// \param parent
  /// \param addBackreference
  void addParent(GraphNode &parent, bool addBackreference = true);

  ///
  /// \return (A reference to) the children nodes of this NodeRelationship.
  std::vector<std::reference_wrapper<GraphNode>> getChildren();

  ///
  /// \return (A const reference to) the children nodes of this NodeRelationship.
  [[nodiscard]] std::vector<std::reference_wrapper<const GraphNode>> getChildren() const;

  ///
  /// \return (A reference to) the parent nodes of this NodeRelationship.
  std::vector<std::reference_wrapper<GraphNode>> getParents();

  ///
  /// \return (A const reference to) the parent nodes of this NodeRelationship.
  [[nodiscard]] std::vector<std::reference_wrapper<const GraphNode>> getParents() const;

  ///
  /// \return
  GraphNode &getOnlyChild();

  ///
  /// \return
  [[nodiscard]] const GraphNode &getOnlyChild() const;

  ///
  /// \param index
  /// \return
  [[nodiscard]] GraphNode &getChildAtIndex(int index);

  ///
  /// \param index
  /// \return
  [[nodiscard]] const GraphNode &getChildAtIndex(int index) const;

  /// Traverses all nodes reachable from the current nodes (descendants) in a depth-first search style and prints the
  /// node's unique node ID.
  /// \param outputStream The stream where to send the output string to.
  void printNodes(std::ostream &outputStream = std::cout) const;

  /// Compares the current graph with another graph given by its root node.
  /// Note: This method does not compare the AST nodes but instead only traverses both graphs (GraphNode objects) and
  /// compares the number of children and parent nodes (graph structure-based comparison).
  /// \param rootNodeOther The root node of the second graph.
  /// \return True if both graphes have the same structure, otherwise False.
  bool isEqualToGraph(GraphNode &rootNodeOther) const;

  /// Iterates over all reachable GraphNodes by considering all children and parent nodes and returns the set of
  /// reachable nodes. Note that it only considers the RelationshipType of this NodeRelationship.
  /// \return The set of nodes that are reachable from this NodeRelationship.
  [[nodiscard]] std::set<std::reference_wrapper<GraphNode>> getAllReachableNodes() const;

  /// Checks whether the given GraphNode node is a child node in this NodeRelationship.
  /// \param node The node to be checked whether it is a child node.
  /// \return True if the given node is a child node in this NodeRelationship.
  bool hasChild(GraphNode &node);

  /// Checks whether the given GraphNode node is a parent node in this NodeRelationship.
  /// \param node The node to be checked whether it is a parent node.
  /// \return True if the given node is a parent node in this NodeRelationship.
  bool hasParent(GraphNode &node);
};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_CONTROLFLOWGRAPH_NODERELATIONSHIP_H_
