#ifndef AST_OPTIMIZER_INCLUDE_AST_ABSTRACTNODE_H_
#define AST_OPTIMIZER_INCLUDE_AST_ABSTRACTNODE_H_

#include <vector>
#include <map>
#include <sstream>
#include <string>
#include <typeinfo>
#include <nlohmann/json.hpp>
#include "Visitor.h"

using json = nlohmann::json;

class AbstractNode {
 protected:
  /// Stores the reserved node ID until the first call of getUniqueNodeId() at which the reserved ID is
  /// fetched and the node's ID is assigned (field uniqueNodeId) based on the node's name and this reserved ID.
  /// This is a workaround because getNodeType() is a virtual method that cannot be called from derived classes'
  /// constructor.
  int assignedNodeId{-1};

  /// Stores the children nodes of the current node.
  std::vector<AbstractNode *> children{};

  /// Stores the parent nodes of the current node.
  std::vector<AbstractNode *> parents{};

  /// A static ongoing counter that is incremented after creating a new AbstractNode object. The counter's value is used
  /// to build the unique node ID.
  static int nodeIdCounter;

  /// An identifier that is unique among all nodes during runtime.
  std::string uniqueNodeId;

  /// Generates a new node ID in the form "<NodeTypeName>_nodeIdCounter" where <NodeTypeName> is the value obtained by
  /// getNodeType() and nodeIdCounter an ongoing counter of created AbstractNode objects.
  /// \return An unique node ID to be used as uniqueNodeId for the current node.
  std::string generateUniqueNodeId();

  /// Returns the current ID (integer) and increments the ID by one. The ID is an ongoing counter of created
  /// AbstractNode objects and is used to build an AbstractNode's unique ID (see getUniqueNodeId()).
  /// \return The current ID as integer.
  static int getAndIncrementNodeId();

  /// Default Constructor, defines some default behavior for subclasses related to IDs
  AbstractNode();

 public:
  /// Virtual Destructor, force class to be abstract
  virtual ~AbstractNode() = 0;

  /// Returns the node's type, which is the name of the object in the AST. This method must be overridden by all classes
  /// that inherit from AbstractNode by their respective name (e.g., ArithmeticExp, Function, Variable).
  /// \return The name of the node type.
  [[nodiscard]] virtual std::string getNodeType() const = 0;

  /// Returns a node's unique ID, or generates it by calling generateUniqueNodeId() if the name was not defined yet.
  /// \return The node's name consisting of the node type and an ongoing number (e.g., Function_1).
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

  /// Returns a vector of pointers to children nodes but without those children that are nullptr.
  /// \return A vector of non-nullptr children.
  [[nodiscard]] std::vector<AbstractNode *> getChildrenNonNull() const;

  /// Returns all the descendants nodes of the current node, i.e., the children of the children and the children of
  /// their children et cetera.
  /// \return A list of descendant nodes.
  std::vector<AbstractNode *> getDescendants();

  /// Adds a new child node to the node's list of children. If addBackReference is True then also updates the child's
  /// list of parent nodes.
  /// \param child The node to be added as child.
  /// \param addBackReference If True, then adds this node as parent to the child node.
  void addChild(AbstractNode *child, bool addBackReference = true);

  /// Adds multiple children to the node's list of children. If addBackReference is True then also updates the child's
  /// list of parent nodes for each of the added children.
  /// \param childrenToAdd A vector of nodes to be added as children to this node.
  /// \param addBackReference If True, then adds this node as parent to each of the child nodes.
  /// otherwise adds the new children at the end of the children vector (append).
  void addChildren(const std::vector<AbstractNode *> &childrenToAdd, bool addBackReference = true);

  void addChildren(const std::vector<AbstractNode *> &childrenToAdd, bool addBackReference,
                   std::vector<AbstractNode *>::const_iterator insertPosition);

  /// Removes the given child from the list of children. If getMaxNumberChildren() returns -1 (i.e., this node supports
  /// an inifinite number of children, then the respective child is simply deleted. In any other case, the child node is
  /// overwritten by -1 such that the order of other children is preserved.
  /// \param child The child to be removed from this node's children.
  /// \param removeBackreference If True, then also removes this node from the children's list of parent nodes.
  void removeChild(AbstractNode *child, bool removeBackreference = true);

  /// Removes all children from this node. Note: Does not update the child's parent.
  void removeChildren();

  /// Replaces a given child (originalChild) of this node by a new node (newChild) and updates the child and
  /// parent references of both nodes. This method preserves the order of the children.
  /// \param originalChild The node to be replaced by newChild.
  /// \param newChild The node to be added at the same position as the original child was.
  virtual void replaceChild(AbstractNode *originalChild, AbstractNode *newChild);

  /// Replaces a given child (originalChild) of this node by one or multiple new nodes (newChildren) and updates the
  /// child and parent references of all affected nodes. This method preserves the order of the children when
  /// inserting the nodes. For example:
  ///     replaceChildren(C, [X, V, W]) on node with children = [a b C d e f]
  /// results in
  ///     children = [a b X V W d e f].
  /// \param originalChild The node to be replaced by newChildren.
  /// \param newChildren The children to be added at the originalChild's current position.
  void replaceChildren(AbstractNode *originalChild, std::vector<AbstractNode *> newChildren);

  /// Returns the number of children nodes that are not null (nullptr).
  /// \return An integer indicating the number of non-nullptr children nodes.
  [[nodiscard]] int countChildrenNonNull() const;

  /// Returns the child at the given index.
  /// \param idx The position of the children in the AbstractNode::children vector.
  /// \return The child at the given index of the children vector, or a nullptr if there is no child at this position.
  [[nodiscard]] AbstractNode *getChildAtIndex(int idx) const;

  /// This special variant of getChildAtIndex returns the n-th parent instead of n-th child if isEdgeDirectionAware is
  /// passed and is true, and the current node has the property isReversed set to True.
  /// \param idx The position of the child to be retrieved.
  /// \param isEdgeDirectionAware If the node's status of isReversed should be considered.
  /// \return A reference to the node at the specified index in the children or parent vector.
  [[nodiscard]] AbstractNode *getChildAtIndex(int idx, bool isEdgeDirectionAware) const;
  /** @} */ // End of children group

  /** @defgroup parents Methods for handling Parents
   *  @{
   */

  /// Returns a reference to the vector of parent nodes.
  /// \return A reference to the vector of this node's parents.
  [[nodiscard]] const std::vector<AbstractNode *> &getParents() const;

  /// Returns a vector of pointers to parent nodes but without those parents that are nullptr.
  /// \return A vector of non-nullptr parent nodes.
  [[nodiscard]] std::vector<AbstractNode *> getParentsNonNull() const;

  /// Returns all the ancestor nodes of the current node, i.e., the ancestors of this node, the ancestors of the
  /// ancestors et cetera.
  /// \return A list of ancestor nodes.
  std::vector<AbstractNode *> getAncestors();

  /// Adds a new parent to this node's list of parents.
  /// \param parentToAdd The parent node to be added.
  /// \param addBackreference If True, then also adds this node as child to the new parent node.
  void addParent(AbstractNode *parentToAdd, bool addBackreference = true);

  /// Removes a certain parent from this node.
  /// \param parentToBeRemoved The node to be removed from this node's parents.
  /// \param removeBackreference If True, then also removes this node from the parentToBeRemoved node's children list.
  void removeParent(AbstractNode *parentToBeRemoved, bool removeBackreference = true);

  /// Removes this node from its parent's children list. If removeParentBackreference is True, then also removes the
  /// parents from this node's parent list.
  /// \param removeParentBackreference Indicates whether to update this node's parents list too.
  AbstractNode *removeFromParents(bool removeParentBackreference = true);

  /// Removes all parents from this node. Note: Does not update the parent's children.
  void removeParents();

  /// Checks whether this node has a certain parent (parentNode).
  /// \param parentNode The node that is searched for in this node's parent list.
  /// \return True if this node has the given parentNode as parent, otherwise returns False.
  bool hasParent(AbstractNode *parentNode);

  /// Returns a pointer to the only parent node. If this node has more than one parent, then a std::logic_exception is
  /// thrown.
  /// \return A pointer to the node's only parent node.
  AbstractNode *getOnlyParent();

  /** @} */ // End of parents group

  /// Swaps the children and parents vectors which corresponds to flipping the edges of this node.
  /// Note: This should never be performed on individual nodes only but instead on the whole AST, see method
  /// Ast::reverseEdges(). Keep in mind that most methods are not aware of this swapped relationship and do not work.
  /// Use *getChildAtIndex(int idx, bool isEdgeDirectionAware) by specifying isEdgeDirectionAware=true to get a specific
  /// child.
  void swapChildrenParents();

  /// Part of the visitor pattern.
  /// Must be overridden in derived classes and must call v.visit(node).
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

  /// Creates a flat copy of the node without including any parents or children.
  /// \return A flat copy of this node.
  [[nodiscard]] virtual AbstractNode *cloneFlat();

  /// Clones a node recursively, i.e., by including all of its children.
  /// \param keepOriginalUniqueNodeId Specifies whether to keep all of the unique node IDs of the original nodes.
  /// \return A clone of the node including clones of all of its children.
  [[nodiscard]] virtual AbstractNode *clone(bool keepOriginalUniqueNodeId) = 0;

  /// Method that updates a cloned node. Must be called within each derived clone() method.
  /// \param keepOriginalUniqueNodeId Determines whether to replace the cloned node's unique node ID by the ID of the
  ///        original node.
  /// \param originalNode The node of which the cloned node is based on.
  void updateClone(bool keepOriginalUniqueNodeId, AbstractNode *originalNode);

  /// Sets the uniqueNodeId attribute. This attribute should be auto-generated by generateUniqueNodeId().
  /// \param newUniqueNodeId The new unique node's identifier.
  void setUniqueNodeId(const std::string &newUniqueNodeId);

  /// This method returns True iff the class derived from the AbstractNode class properly makes use of the child/parent
  /// fields as it would be expected in a circuit.
  virtual bool supportsCircuitMode();

  /// Indicates whether the edges of this node are reversed compared to its initial state.
  bool isReversed{false};

  /// Removes this node from all of its parents and children, and also removes all parents and children from this node.
  void isolateNode();

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
      outputMsg << this->getNodeType() << ".";
      throw std::logic_error(outputMsg.str());
    }
  }

  /// Generates an output string to be used by the toString() method.
  /// \param printChildren Specifies whether to print details of this node only (False) or also its children (True).
  /// \param attributes The node's attributes to be printed (fields in the node's class).
  /// \return A string representation of the node.
  [[nodiscard]] std::string generateOutputString(bool printChildren, std::vector<std::string> attributes) const;
};

#endif //AST_OPTIMIZER_INCLUDE_AST_ABSTRACTNODE_H_
