#ifndef MASTER_THESIS_CODE_NODE_H
#define MASTER_THESIS_CODE_NODE_H

#include <vector>
#include <nlohmann/json.hpp>
#include "../include/visitor/Visitor.h"

using json = nlohmann::json;

class Literal;

class Ast;

class Node {
 protected:
  std::vector<Node*> children{};

  std::vector<Node*> parents{};

  static int nodeIdCounter;

  /// An identifier that is unique among all nodes during runtime.
  std::string uniqueNodeId;

  /// This attributes is used to link back to the original Node in an overlay circuit.
  Node* underlyingNode{};

  /// Generates a new node ID in the form "<NodeTypeName>_nodeIdCounter++" where <NodeTypeName> is the value obtained by
  /// getNodeName() and nodeIdCounter an ongoing counter of created Node objects.
  /// \return An unique node ID to be used as uniqueNodeId for the current node.
  std::string genUniqueNodeId();

  static int getAndIncrementNodeId();

  /// This special variant of getChildAtIndex returns the n-th parent instead of n-th child if isEdgeDirectionAware is
  /// passed and is true, and the current node has the property isReversed set to True.
  /// \param idx The position of the child to be retrieved.
  /// \param isEdgeDirectionAware If the node's status of isReversed should be considered.
  /// \return A reference to the node at the specified index in the children or parent vector.
  [[nodiscard]] Node* getChildAtIndex(int idx, bool isEdgeDirectionAware) const;

 public:
  Node();

  [[nodiscard]] Node* getUnderlyingNode() const;
  void setUnderlyingNode(Node* uNode);

  [[nodiscard]] virtual std::string getNodeName() const;

  std::string getUniqueNodeId();

  static void resetNodeIdCounter();

  [[nodiscard]] const std::vector<Node*> &getParents() const;

  [[nodiscard]] const std::vector<Node*> &getChildren() const;

  /// Returns all the ancestor nodes of the current node.
  /// \return A list of ancestor nodes.
  std::vector<Node*> getAnc();

  // Functions for handling children
  void addChild(Node* child, bool addBackReference = false);
  void addChildBilateral(Node* child);
  void addChildren(const std::vector<Node*> &childrenToAdd, bool addBackReference = false);
  void setChild(std::__wrap_iter<Node* const*> position, Node* value);
  void removeChild(Node* child);
  void removeChildren();
  [[nodiscard]] int countChildrenNonNull() const;

  /// Returns the child at the given index.
  /// \param idx The position of the children in the Node::children vector.
  /// \return The child at the given index of the children vector, or a nullptr if there is no child at this position.
  [[nodiscard]] Node* getChildAtIndex(int idx) const;

  // Functions for handling parents
  void addParent(Node* n);
  void removeParent(Node* node);
  void removeParents();
  bool hasParent(Node* n);

  static void addParentTo(Node* parentNode, std::vector<Node*> nodesToAddParentTo);

  void swapChildrenParents();

  virtual Literal* evaluate(Ast &ast);

  virtual void accept(Visitor &v);

  [[nodiscard]] virtual json toJson() const;

  [[nodiscard]] virtual std::string toString() const;

  friend std::ostream &operator<<(std::ostream &os, const std::vector<Node*> &v);

  [[nodiscard]] virtual Node* clone();

  [[nodiscard]] virtual Node* cloneRecursiveDeep();

  void setUniqueNodeId(const std::string &unique_node_id);

  /// Determine the value of this node for computing the multiplicative depth and reverse multiplicative depth,
  /// getMultDepthL() and getReverseMultDepthR(), respectively.
  /// \return Returns 1 iff this node is a LogicalExpr containing an AND operator, otherwise 0.
  int depthValue();

  /// Calculates the multiplicative depth based on the definition given in
  /// [Aubry, P. et al.: Faster Homomorphic Encryption Is Not Enough: Improved Heuristic for Multiplicative Depth
  ///  Minimization of Boolean Circuits. (2019)].
  /// \return The multiplicative depth of the current node.
  int getMultDepthL(std::map<std::string, int>* storedDepthsMap = nullptr);

  /// Calculates the reverse multiplicative depth based on the definition given in
  /// [Aubry, P. et al.: Faster Homomorphic Encryption Is Not Enough: Improved Heuristic for Multiplicative Depth
  ///  Minimization of Boolean Circuits. (2019)].
  /// \return The reverse multiplicative depth of the current node.
  int getReverseMultDepthR(std::map<std::string, int>* storedDepthsMap = nullptr);

  /// This method returns True iff the class derived from the Node class properly makes use of the child/parent fields
  /// as it would be expected in a circuit.
  virtual bool supportsCircuitMode();

  /// Indicates the number of children that are allowed for a specific node.
  /// For example, a binary expression accepts exactly three attributes and hence also exactly three children:
  /// left operand, right operand, and operator.
  /// If the node does not implement support for child/parent relationships, getMaxNumberChildren() return 0.
  /// \return An integer indicating the number of allowed children for a specific node.
  virtual int getMaxNumberChildren();

  /// Indicates whether the edges of this node are reversed compared to its initial state.
  bool isReversed{false};
  std::vector<Node*> getChildrenNonNull() const;
  std::vector<Node*> getParentsNonNull() const;
  void isolateNode();
  void removeChildBilateral(Node* child);
  virtual ~Node();

  bool hasReversedEdges() const;
};

#endif //MASTER_THESIS_CODE_NODE_H
