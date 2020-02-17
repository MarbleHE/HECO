#ifndef AST_OPTIMIZER_INCLUDE_NODE_H
#define AST_OPTIMIZER_INCLUDE_NODE_H

#include <vector>
#include <map>
#include <sstream>
#include <string>
#include <nlohmann/json.hpp>
#include "Visitor.h"
#include "OpSymbEnum.h"

using json = nlohmann::json;

class Literal;

class Ast;

class Node {
 private:
  [[nodiscard]] virtual Node *createClonedNode(bool keepOriginalUniqueNodeId);

 protected:
  /// Temporarily stores the reserved node ID until the first call of getUniqueNodeId() at which the reserved ID is
  /// fetched and the node's ID is assigned (field uniqueNodeId) based on the node's name and this reserved ID.
  /// This is a workaround because getNodeName() is a virtual method that cannot be called from derived classes and
  /// their constructor. After retrieving the node ID and assigning it to the uniqueNodeId field, it is deleted from
  /// this map.
  std::map<Node *, int> assignedNodeIds{};

  /// Stores the children of the current node if the node supports the circuit mode (see supportsCircuitMode()).
  std::vector<Node *> children{};

  /// Stores the parent nodes of the current node if the node supports the circuit mode (see supportsCircuitMode()).
  std::vector<Node *> parents{};

  /// A static ongoing counter that is incremented after creating a new Node object. The counter's value is used to
  /// build the unique node ID.
  static int nodeIdCounter;

  /// An identifier that is unique among all nodes during runtime.
  std::string uniqueNodeId;

  /// This attributes is used to link back to the original Node if this node is part of an overlay circuit representing
  /// only a subset of certain nodes. Required, for example, by cone rewriting.
  Node *underlyingNode{};

  /// Generates a new node ID in the form "<NodeTypeName>_nodeIdCounter++" where <NodeTypeName> is the value obtained by
  /// getNodeName() and nodeIdCounter an ongoing counter of created Node objects.
  /// \return An unique node ID to be used as uniqueNodeId for the current node.
  std::string genUniqueNodeId();

  ///
  /// \return
  static int getAndIncrementNodeId();

  /// This special variant of getChildAtIndex returns the n-th parent instead of n-th child if isEdgeDirectionAware is
  /// passed and is true, and the current node has the property isReversed set to True.
  /// \param idx The position of the child to be retrieved.
  /// \param isEdgeDirectionAware If the node's status of isReversed should be considered.
  /// \return A reference to the node at the specified index in the children or parent vector.
  [[nodiscard]] Node *getChildAtIndex(int idx, bool isEdgeDirectionAware) const;

 public:
  Node();

  virtual ~Node();

  [[nodiscard]] Node *getUnderlyingNode() const;

  void setUnderlyingNode(Node *uNode);

  [[nodiscard]] virtual std::string getNodeName() const;

  std::string getUniqueNodeId();

  /// Resets the static node ID counter that is used to build the unique node ID. This method is required for testing.
  static void resetNodeIdCounter();

  /// Returns a reference to the vector of parent nodes.
  /// \return A reference to the vector of this node's parents.
  [[nodiscard]] const std::vector<Node *> &getParents() const;

  /// Returns a reference to the vector of children nodes.
  /// \return A reference to the vector of this node's children.
  [[nodiscard]] const std::vector<Node *> &getChildren() const;

  /// Returns all the ancestor nodes of the current node, i.e., the ancestors of this node, the ancestors of the
  /// ancestors et cetera.
  /// \return A list of ancestor nodes.
  std::vector<Node *> getAncestors();

  /// Returns all the descendants nodes of the current node, i.e., the children of the children and the children of
  /// their children et cetera.
  /// \return A list of descendant nodes.
  std::vector<Node *> getDescendants();

  // Functions for handling children
  void addChild(Node *child, bool addBackReference = false);

  void addChildBilateral(Node *child);

  void addChildren(const std::vector<Node *> &childrenToAdd, bool addBackReference = false);

  void setChild(std::vector<Node *>::const_iterator position, Node *value);

  void removeChild(Node *child);

  void removeChildren();

  [[nodiscard]] int countChildrenNonNull() const;

  /// Returns the child at the given index.
  /// \param idx The position of the children in the Node::children vector.
  /// \return The child at the given index of the children vector, or a nullptr if there is no child at this position.
  [[nodiscard]] Node *getChildAtIndex(int idx) const;

  // Functions for handling parents
  void addParent(Node *n);

  void removeParent(Node *node);

  void removeParents();

  bool hasParent(Node *n);

  static void addParentTo(Node *parentNode, std::vector<Node *> nodesToAddParentTo);

  void swapChildrenParents();

  virtual std::vector<Literal *> evaluate(Ast &ast);

  virtual void accept(Visitor &v);

  [[nodiscard]] virtual json toJson() const;

  [[nodiscard]] virtual std::string toString() const;

  friend std::ostream &operator<<(std::ostream &os, const std::vector<Node *> &v);

  [[nodiscard]] virtual Node *cloneFlat();

  void setUniqueNodeId(const std::string &unique_node_id);

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

  [[nodiscard]] std::vector<Node *> getChildrenNonNull() const;

  [[nodiscard]] std::vector<Node *> getParentsNonNull() const;

  /// Removes this node from all of its parents and children, and also removes all parents and children from this node.
  void isolateNode();

  /// Removes the node 'child' bilateral, i.e., on both ends of the edge. In other words, removes the node 'child' from
  /// this node, and this node from the parents list of 'child' node.
  /// \param child The child to be removed from this node.
  void removeChildBilateral(Node *child);

  /// Checks whether the edges of this node are reversed (i.e., node's parents and children are swapped).
  /// \return True iff the node's edges are reversed.
  [[nodiscard]] bool hasReversedEdges() const;

  /// Transforms a multi-input gate taking N inputs into a sequence of binary gates.
  ///
  /// For example, consider a N input logical-AND (&) with inputs y_1 to y_m:
  ///   <pre> &_{i=1}^{n} y_1, y_2, y_3, ..., y_m. </pre>
  /// It is transformed by this method into the expression:
  ///   <pre> ((((y_1 & y_2) & y_3) ...) & y_m), </pre>
  /// wherein each AND-gate only has two inputs (binary gates).

  /// \param inputNodes The inputs y_1, ..., y_m that are connected to the multi-input gate. It is required that m>=2.
  /// \param gateType The gate that all inputs are connected to.
  /// \return A vector of Node objects of type LogicalExpr that represent the chain of LogicalExpr required to represent
  /// the intended multi-input gate. The last node in inputNodes (i.e., inputNodes.back()) is always the output of this
  /// chain.
  static std::vector<Node *> rewriteMultiInputGateToBinaryGatesChain(std::vector<Node *> inputNodes,
                                                                     OpSymb::LogCompOp gateType);

  /// Casts a node to type T which must be the specific derived class of the node to cast successfully.
  /// \tparam T The derived class of the node object.
  /// \return A pointer to the casted object, or a std::logic_error if cast was unsuccessful.
  template<typename T>
  T *castTo() {
    if (auto castedNode = dynamic_cast<T *>(this)) {
      return castedNode;
    } else {
      std::stringstream outputMsg;
      outputMsg << "Cannot cast object of type Node to given class ";
      outputMsg << typeid(T).name() << ". ";
      outputMsg << "Because node (" << this->getUniqueNodeId() << ") is of type ";
      outputMsg << this->getNodeName() << ".";
      throw std::logic_error(outputMsg.str());
    }
  }

  bool hasChild(Node *n);

  Node *cloneRecursiveDeep(bool keepOriginalUniqueNodeId);

  Literal *ensureSingleEvaluationResult(std::vector<Literal *> evaluationResult);
};

#endif //AST_OPTIMIZER_INCLUDE_NODE_H
