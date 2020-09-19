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
  ///
  RelationshipType relationshipType;

  ///
  GraphNode &graphNode;

  ///
  std::vector<std::reference_wrapper<GraphNode>> children;

  ///
  std::vector<std::reference_wrapper<GraphNode>> parents;

 public:
  ///
  /// \param relationshipType
  /// \param graphNode
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
  /// \return
  std::vector<std::reference_wrapper<GraphNode>> getChildren();

  ///
  /// \return
  [[nodiscard]] std::vector<std::reference_wrapper<const GraphNode>> getChildren() const;

  ///
  /// \return
  std::vector<std::reference_wrapper<GraphNode>> getParents();

  ///
  /// \return
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

//  /// Compares two graphs given by their respective root node. The node on that this method is called is treated as the
//  /// first root node, the second root node needs to be passed as rootNodeOther.
//  /// Note: This method does not compare the refToOriginalNode but instead only  traverses both graphs and compares the
//  /// number of children and parent nodes (structure-based comparison).
//  /// \param rootNodeOther The root node of the second graph.
//  /// \return True if both graphes have the same structure, otherwise False.
//  bool areEqualGraphs(GraphNode *rootNodeOther) const;

  ///
  /// \return
  [[nodiscard]] std::set<std::reference_wrapper<GraphNode>> getAllReachableNodes() const;

  ///
  /// \param node
  /// \return
  bool hasChild(GraphNode &node);

  ///
  /// \param node
  /// \return
  bool hasParent(GraphNode &node);
};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_CONTROLFLOWGRAPH_NODERELATIONSHIP_H_
