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
  Node* underlyingNode;

  /// Generates a new node ID in the form "<NodeTypeName>_nodeIdCounter++" where <NodeTypeName> is the value obtained by
  /// getNodeName() and nodeIdCounter an ongoing counter of created Node objects.
  /// \return An unique node ID to be used as uniqueNodeId for the current node.
  std::string genUniqueNodeId();

  static int getAndIncrementNodeId();

  static int getNodeIdCounter();

 public:
  Node();

  [[nodiscard]] Node* getUnderlyingNode() const;
  void setUnderlyingNode(Node* uNode);

  [[nodiscard]] virtual std::string getNodeName() const;

  std::string getUniqueNodeId();

  static void resetNodeIdCounter();

  [[nodiscard]] const std::vector<Node*> &getPred() const;
  [[nodiscard]] const std::vector<Node*> &getSucc() const;

  // Functions for handling children
  void addChild(Node* child, bool addBackReference = false);
  void addChildBilateral(Node* child);
  virtual void addChildren(const std::vector<Node*> &childrenToAdd, bool addBackReference = false);
  void setChild(std::__wrap_iter<Node* const*> position, Node* value);
  void removeChild(Node* child);
  void removeChildren();
  [[nodiscard]] const std::vector<Node*> &getChildren() const;
  [[nodiscard]] std::vector<Node*> getChildrenNonNull() const;

  // Functions for handling parents
  void addParent(Node* n);
  void removeParent(Node* node);
  void removeParents();
  [[nodiscard]] const std::vector<Node*> &getParents() const;
  bool hasParent(Node* n);

  static void addParentTo(Node* parentNode, std::vector<Node*> nodesToAddParentTo);

  void swapChildrenParents();

  virtual Literal* evaluate(Ast &ast);

  virtual void accept(Visitor &v);

  [[nodiscard]] virtual json toJson() const;

  [[nodiscard]] virtual std::string toString() const;

  friend std::ostream &operator<<(std::ostream &os, const std::vector<Node*> &v);

  std::string getDotFormattedString(bool isReversed, const std::string &indentation, bool showMultDepth);

  [[nodiscard]] virtual Node* clone();

  [[nodiscard]] virtual Node* cloneRecursiveDeep();

  void setUniqueNodeId(const std::string &unique_node_id);

  std::vector<Node*> getAnc();

  int depthValue();

  int getMultDepthL();

  int getReverseMultDepthR();

/// This method should return True iff the class derived from Node class does properly use the child/parent fields as
/// it would be expected in a circuit.
  virtual bool supportsCircuitMode();
  Node* getChildAtIndex(int idx) const;
/// Indicates the number of children that are allowed for a specific node.
/// For example, a binary expression accepts exactly three attributes and hence also exactly three children:
/// left operand, right operand, and operator.
/// \return -1 iff the number of the node's children is unlimited, otherwise an integer number.
  virtual int getMaxNumberChildren();
};

#endif //MASTER_THESIS_CODE_NODE_H
