#ifndef AST_OPTIMIZER_INCLUDE_UTILITIES_NODERELATIONSHIP_H_
#define AST_OPTIMIZER_INCLUDE_UTILITIES_NODERELATIONSHIP_H_

#include <vector>
#include <iostream>

enum class RelationshipType { CTRL_FLOW_GRAPH, DATA_FLOW_GRAPH };

// forward declarations
class GraphNode;

class NodeRelationship {
 private:
  RelationshipType rship;

  GraphNode *refToGraphNode;

  std::vector<GraphNode *> children;
  std::vector<GraphNode *> parents;

 public:
  NodeRelationship(RelationshipType relationship, GraphNode *gNode);

  void addChild(GraphNode *child, bool addBackreference = true);

  void addParent(GraphNode *parent, bool addBackreference = true);

  [[nodiscard]] const std::vector<GraphNode *> &getChildren() const;

  [[nodiscard]] const std::vector<GraphNode *> &getParents() const;

  [[nodiscard]] GraphNode *getOnlyChild() const;

  [[nodiscard]] GraphNode *getChildAtIndex(int idx) const;

  /// Traverses all nodes reachable from the current nodes (descendants) in a depth-first search style and prints the
  /// node's unique node ID.
  /// \param outputStream The stream where to send the output string to.
  void traverseAndPrintNodes(std::ostream &outputStream = std::cout) const;

  /// Compares two graphs given by their respective root node. The node on that this method is called is treated as the
  /// first root node, the second root node needs to be passed as rootNodeOther.
  /// Note: This method does not compare the refToOriginalNode but instead only  traverses both graphs and compares the
  /// number of children and parent nodes (structure-based comparison).
  /// \param rootNodeOther The root node of the second graph.
  /// \return True if both graphes have the same structure, otherwise False.
  bool areEqualGraphs(GraphNode *rootNodeOther) const;
};

#endif //AST_OPTIMIZER_INCLUDE_UTILITIES_NODERELATIONSHIP_H_
