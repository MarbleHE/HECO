#ifndef AST_OPTIMIZER_INCLUDE_AST_NODE_H_
#define AST_OPTIMIZER_INCLUDE_AST_NODE_H_

#include <vector>
#include <map>
#include <sstream>
#include <string>
#include <nlohmann/json.hpp>
#include "Visitor.h"
#include <typeinfo>

using json = nlohmann::json;

class AbstractNode {
 protected:
  /// Stores the reserved node ID until the first call of getUniqueNodeId() at which the reserved ID is
  /// fetched and the node's ID is assigned (field uniqueNodeId) based on the node's name and this reserved ID.
  /// This is a workaround because getNodeName() is a virtual method that cannot be called from derived classes'
  /// constructor.
  int assignedNodeId{-1};

  /// Stores the children of the current node.
  std::vector<AbstractNode *> children{};

  /// Stores the parent nodes of the current node.
  std::vector<AbstractNode *> parents{};

  /// A static ongoing counter that is incremented after creating a new AbstractNode object. The counter's value is used to
  /// build the unique node ID.
  static int nodeIdCounter;

  /// An identifier that is unique among all nodes during runtime.
  std::string uniqueNodeId;

  /// Generates a new node ID in the form "<NodeTypeName>_nodeIdCounter++" where <NodeTypeName> is the value obtained by
  /// getNodeName() and nodeIdCounter an ongoing counter of created AbstractNode objects.
  /// \return An unique node ID to be used as uniqueNodeId for the current node.
  std::string genUniqueNodeId();

  /// Returns the current ID (integer) and increments the ID by one. The ID is an ongoing counter of created
  /// AbstractNode objects and is used to build an AbstractNode's unique ID (see getUniqueNodeId()).
  /// \return The current ID as integer.
  static int getAndIncrementNodeId();

  /// This special variant of getChildAtIndex returns the n-th parent instead of n-th child if isEdgeDirectionAware is
  /// passed and is true, and the current node has the property isReversed set to True.
  /// \param idx The position of the child to be retrieved.
  /// \param isEdgeDirectionAware If the node's status of isReversed should be considered.
  /// \return A reference to the node at the specified index in the children or parent vector.
  [[nodiscard]] AbstractNode *getChildAtIndex(int idx, bool isEdgeDirectionAware) const;

  /// Default Constructor, defines some default behavior for subclasses related to IDs
  AbstractNode();

 public:
  /// Virtual Destructor, force class to be abstract
  virtual ~AbstractNode() = 0;

  [[nodiscard]] virtual std::string getNodeName() const = 0;

  std::string getUniqueNodeId();

  /// Resets the static node ID counter that is used to build the unique node ID.
  /// This method is required for testing.
  static void resetNodeIdCounter();


  /** @defgroup parents Methods for handling children
   *  @{
   */

  /// Indicates the number of children that are allowed for a specific node.
  /// For example, a arithmetic expression accepts exactly three attributes and hence also exactly three children:
  /// left operand, right operand, and operator.
  /// If the node does not implement support for child/parent relationships, getMaxNumberChildren() return 0.
  /// \return An integer indicating the number of allowed children for a specific node.
  virtual int getMaxNumberChildren();

  /// Returns a reference to the vector of children nodes.
  /// \return A reference to the vector of this node's children.
  [[nodiscard]] const std::vector<AbstractNode *> &getChildren() const;

  [[nodiscard]] std::vector<AbstractNode *> getChildrenNonNull() const;

  /// Returns all the descendants nodes of the current node, i.e., the children of the children and the children of
  /// their children et cetera.
  /// \return A list of descendant nodes.
  std::vector<AbstractNode *> getDescendants();

  void addChild(AbstractNode *child, bool addBackReference = true);

  void addChildren(const std::vector<AbstractNode *> &childrenToAdd, bool addBackReference = true);

  void setChild(std::vector<AbstractNode *>::const_iterator position, AbstractNode *value);

  void removeChild(AbstractNode *child);

  void removeChildren();

  void replaceChild(AbstractNode *originalChild, AbstractNode *newChildToBeAdded);

  [[nodiscard]] int countChildrenNonNull() const;

  /// Returns the child at the given index.
  /// \param idx The position of the children in the AbstractNode::children vector.
  /// \return The child at the given index of the children vector, or a nullptr if there is no child at this position.
  [[nodiscard]] AbstractNode *getChildAtIndex(int idx) const;
  /** @} */ // End of children group

  /** @defgroup parents Methods for handling Parents
   *  @{
   */

  /// Returns a reference to the vector of parent nodes.
  /// \return A reference to the vector of this node's parents.
  [[nodiscard]] const std::vector<AbstractNode *> &getParents() const;

  [[nodiscard]] std::vector<AbstractNode *> getParentsNonNull() const;

  /// Returns all the ancestor nodes of the current node, i.e., the ancestors of this node, the ancestors of the
  /// ancestors et cetera.
  /// \return A list of ancestor nodes.
  std::vector<AbstractNode *> getAncestors();

  void addParent(AbstractNode *n);

  void removeParent(AbstractNode *node);

  void removeFromParents(bool removeParentBackreference = true);

  void removeParents();

  bool hasParent(AbstractNode *n);

  void swapChildrenParents();
  /** @} */ // End of parents group

  /// Part of the visitor pattern.
  /// Must be overriden in derived classes and must call v.visit(node).
  /// This allows the correct overload for the derived class to be called in the visitor.
  /// \param v Visitor that offers a visit() method
  virtual void accept(Visitor &v) = 0;

  /// Get the JSON representation of the node including all of its children.
  /// \return JSON representation of the node
  [[nodiscard]] virtual json toJson() const;

  /// Returns a string representation of the node, if printChildren is True then calls toString for its children too.
  /// Hence, toString(false) is equivalent to printing a node's attributes only.
  /// \return A textual representation of the node.
  [[nodiscard]] virtual std::string toString(bool printChildren) const;

  [[nodiscard]] virtual AbstractNode *cloneFlat();

  [[nodiscard]] virtual AbstractNode *clone(bool keepOriginalUniqueNodeId) = 0;

  void setUniqueNodeId(const std::string &newUniqueNodeId);

  /// This method returns True iff the class derived from the AbstractNode class properly makes use of the child/parent fields
  /// as it would be expected in a circuit.
  virtual bool supportsCircuitMode();

  /// Indicates whether the edges of this node are reversed compared to its initial state.
  bool isReversed{false};

  /// Removes this node from all of its parents and children, and also removes all parents and children from this node.
  void isolateNode();

  /// Removes the node 'child' bilateral, i.e., on both ends of the edge. In other words, removes the node 'child' from
  /// this node, and this node from the parents list of 'child' node.
  /// \param child The child to be removed from this node.
  void removeChildBilateral(AbstractNode *child);

  /// Checks whether the edges of this node are reversed (i.e., node's parents and children are swapped).
  /// \return True iff the node's edges are reversed.
  [[nodiscard]] bool hasReversedEdges() const;

  /// Casts a node to type T which must be the specific derived class of the node to cast successfully.
  /// \tparam T The derived class of the node object.
  /// \return A pointer to the casted object, or a std::logic_error if cast was unsuccessful.
  template<typename T>
  T *castTo() {
    if (auto castedNode = dynamic_cast<T *>(this)) {
      return castedNode;
    } else {
      std::stringstream outputMsg;
      outputMsg << "Cannot cast object of type AbstractNode to given class ";
      outputMsg << typeid(T).name() << ". ";
      outputMsg << "Because node (" << this->getUniqueNodeId() << ") is of type ";
      outputMsg << this->getNodeName() << ".";
      throw std::logic_error(outputMsg.str());
    }
  }
  std::string generateOutputString(bool printChildren, std::vector<std::string> attributes) const;
};

#endif //AST_OPTIMIZER_INCLUDE_AST_NODE_H_
