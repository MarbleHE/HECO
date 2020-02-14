#include "../include/ast/Node.h"
#include <sstream>
#include <Operator.h>
#include <set>
#include "ConeRewriter.h"
#include "AbstractExpr.h"
#include "Node.h"
#include "LogicalExpr.h"
#include <queue>
#include "Function.h"

int Node::nodeIdCounter = 0;

std::string Node::genUniqueNodeId() {
  int nodeNo;
  try {
    nodeNo = assignedNodeIds.at(this);
  } catch (std::out_of_range) {
    throw std::logic_error(
        "Could not find any reserved ID for node. Node constructor needs to reserve ID for node (see empty constructor).");
  }

  // clear the node entry as we will save the node ID in the uniqueNodeId field
  assignedNodeIds.erase(this);

  // build and return the node ID string
  std::stringstream ss;
  ss << getNodeName() << "_" << nodeNo;
  return ss.str();
}

Node::Node() {
  // save the ID reserved for this node but do not
  assignedNodeIds[this] = getAndIncrementNodeId();
}

std::string Node::getUniqueNodeId() {
  // if there is no ID defined yet, create and assign an ID
  if (uniqueNodeId.empty()) this->uniqueNodeId = genUniqueNodeId();
  // otherwise just return the previously generated ID
  return uniqueNodeId;
}

int Node::getAndIncrementNodeId() {
  return nodeIdCounter++;
}

std::string Node::getNodeName() const {
  return "Node";
}

void Node::resetNodeIdCounter() {
  Node::nodeIdCounter = 0;
}

const std::vector<Node*> &Node::getChildren() const {
  return children;
}

std::vector<Node*> Node::getChildrenNonNull() const {
  std::vector<Node*> childrenFiltered;
  std::copy_if(children.begin(), children.end(), std::back_inserter(childrenFiltered),
               [](Node* n) { return n != nullptr; });
  return childrenFiltered;
}

std::vector<Node*> Node::getParentsNonNull() const {
  std::vector<Node*> parentsFiltered;
  std::copy_if(parents.begin(), parents.end(), std::back_inserter(parentsFiltered),
               [](Node* n) { return n != nullptr; });
  return parentsFiltered;
}

void Node::addChildBilateral(Node* child) {
  addChild(child, true);
}

void Node::addChild(Node* child, bool addBackReference) {
  addChildren({child}, addBackReference);
}

void Node::addChildren(const std::vector<Node*> &childrenToAdd, bool addBackReference) {
  // check whether the number of children to be added does not exceed the number of maximum supported children
  if (childrenToAdd.size() > getMaxNumberChildren()) {
    throw std::invalid_argument("Node " + getUniqueNodeId() + " of type " + getNodeName() + " does not allow more than "
                                    + std::to_string(getMaxNumberChildren()) + " children!");
  }

  // check if circuit mode is supported by current node, otherwise addChildren will lead to unexpected behavior
  if (!this->supportsCircuitMode()) {
    throw std::logic_error(
        "Cannot use addChildren because node does not support circuit mode!");
  }

  // these actions are to be performed after a node was added to the list of children
  auto doInsertPostAction = [&](Node* childToAdd) {
    // if option 'addBackReference' is true, we add a back reference to the child as parent
    if (addBackReference) childToAdd->addParent(this);
  };

  if (getChildren().empty()) {  // if the list of children is still empty, we can simply add all nodes in one batch
    // add children to the vector's end
    children.insert(children.end(), childrenToAdd.begin(), childrenToAdd.end());
    std::for_each(children.begin(), children.end(), doInsertPostAction);
    // if this nodes accepts an infinite number of children, pre-filling the slots does not make any sense -> skip it
    // fill remaining slots with nullptr values
    children.insert(children.end(), getMaxNumberChildren() - getChildren().size(), nullptr);
  } else {  // otherwise we need to add the children one-by-one by looking for free slots
    int childIdx = 0;
    // add child in first empty spot
    for (auto it = getChildren().begin(); it != getChildren().end() && childIdx < childrenToAdd.size(); ++it) {
      if (*it == nullptr) {
        auto childToAdd = childrenToAdd.at(childIdx);
        setChild(it, childToAdd);
        doInsertPostAction(childToAdd);
        childIdx++;
      }
    }
    // check if we were able to add all children, otherwise throw an exception
    if (childIdx != childrenToAdd.size()) {
      throw std::logic_error("Cannot add one or multiple children to " + this->getUniqueNodeId()
                                 + " without overwriting an existing one. Consider removing an existing child first.");
    }
  }
}

void Node::setChild(std::__wrap_iter<Node* const*> position, Node* value) {
  auto newIterator = children.insert(position, value);
  children.erase(++newIterator);
}

void Node::removeChild(Node* child) {
  auto it = std::find(children.begin(), children.end(), child);
  if (it != children.end()) *it = nullptr; //children.erase(it);
}

void Node::removeChildBilateral(Node* child) {
  child->removeParent(this);
  this->removeChild(child);
}

void Node::isolateNode() {
  for (auto &p : getParentsNonNull()) p->removeChild(this);
  for (auto &c : getChildrenNonNull()) c->removeParent(this);
  removeChildren();
  removeParents();
}

const std::vector<Node*> &Node::getParents() const {
  return parents;
}

void Node::addParent(Node* n) {
  parents.push_back(n);
}

void Node::removeParent(Node* parent) {
  auto it = std::find(parents.begin(), parents.end(), parent);
  if (it != parents.end()) parents.erase(it);
}

void Node::removeChildren() {
  children.clear();
}

void Node::removeParents() {
  parents.clear();
}

void Node::addParentTo(Node* parentNode, std::vector<Node*> nodesToAddParentTo) {
  std::for_each(nodesToAddParentTo.begin(), nodesToAddParentTo.end(), [&](Node* n) {
    if (n != nullptr) n->addParent(parentNode);
  });
}

void Node::swapChildrenParents() {
  std::vector<Node*> oldParents = this->parents;
  this->parents = this->children;
  this->children = oldParents;
  // toggle the isReversed boolean
  isReversed = !isReversed;
}

Literal* Node::evaluate(Ast &ast) {
  return nullptr;
}

void Node::accept(Visitor &v) {
  std::cout << "This shouldn't be executed!" << std::endl;
}

void to_json(json &j, const Node &n) {
  j = n.toJson();
}

json Node::toJson() const {
  return json({"type", "Node"});
}

std::string Node::toString() const {
  return this->toJson().dump();
}

std::ostream &operator<<(std::ostream &os, const std::vector<Node*> &v) {
  os << "[";
  for (int i = 0; i < v.size(); ++i) {
    os << v[i]->getUniqueNodeId();
    if (i != v.size() - 1)
      os << ", ";
  }
  os << "]";
  return os;
}

Node* Node::getUnderlyingNode() const {
  return underlyingNode;
}

void Node::setUnderlyingNode(Node* uNode) {
  underlyingNode = uNode;
}

void Node::setUniqueNodeId(const std::string &unique_node_id) {
  uniqueNodeId = unique_node_id;
}

std::vector<Node*> Node::getAnc() {
  // use a set to avoid duplicates as there may be common ancestors between this node and any of the node's parents
  std::set<Node*> result;
  std::queue<Node*> processQueue{{this}};
  while (!processQueue.empty()) {
    auto curNode = processQueue.front();
    processQueue.pop();
    auto nextNodes = curNode->getParents();
    std::for_each(nextNodes.begin(), nextNodes.end(), [&](Node* node) {
      result.insert(node);
      processQueue.push(node);
    });
  }
  return std::vector<Node*>(result.begin(), result.end());
}

Node* Node::cloneFlat() {
  throw std::logic_error("ERROR: cloneFlat() not implemented for node of type " + getNodeName());
}

Node* Node::cloneRecursiveDeep(bool keepOriginalUniqueNodeId) {
  // call polymorphic createClonedNode to copy derived class-specific fields
  Node* clonedNode = this->createClonedNode(keepOriginalUniqueNodeId);

  // perform cloning of fields belonging to Node
  if (keepOriginalUniqueNodeId) clonedNode->setUniqueNodeId(this->getUniqueNodeId());
  if (this->isReversed) clonedNode->swapChildrenParents();
  if (this->underlyingNode != nullptr) clonedNode->setUnderlyingNode(this->getUnderlyingNode());

  return clonedNode;
}

Node* Node::createClonedNode(bool keepOriginalUniqueNodeId) {
  throw std::logic_error(
      "ERROR: Cannot execute cloneRecursiveDeep(...) because createClonedNode(...) is not implemented for node of type "
          + getNodeName());
}

bool Node::hasParent(Node* n) {
  return std::any_of(getParents().begin(), getParents().end(), [&n](Node* p) { return (p == n); });
}

bool Node::hasChild(Node* n) {
  return std::any_of(getChildren().begin(), getChildren().end(), [&n](Node* p) { return (p == n); });
}

int Node::countChildrenNonNull() const {
  return std::count_if(getChildren().begin(), getChildren().end(), [](Node* n) { return n != nullptr; });
}

int Node::getMaxNumberChildren() {
  return 0;
}

bool Node::supportsCircuitMode() {
  return false;
}

Node* Node::getChildAtIndex(int idx) const {
  return getChildAtIndex(idx, false);
}

Node* Node::getChildAtIndex(int idx, bool isEdgeDirectionAware) const {
  try {
    if (isEdgeDirectionAware && isReversed) return parents.at(idx);
    else return children.at(idx);
  } catch (std::out_of_range const &e) {
    return nullptr;
  }
}

Node::~Node() = default;

bool Node::hasReversedEdges() const {
  return isReversed;
}

std::vector<Node*> Node::rewriteMultiInputGateToBinaryGatesChain(std::vector<Node*> inputNodes,
                                                                 OpSymb::LogCompOp gateType) {
  if (inputNodes.empty()) {
    throw std::invalid_argument("Cannot construct a 0-input logical gate!");
  }

  // if there is only one input, we need to add the "neutral element" (i.e., the element that does not change the
  // semantics of the logical expression) depending on the given LogCompOp to inputNodes
  if (inputNodes.size() == 1) {
    if (gateType == OpSymb::LogCompOp::logicalXor) {
      // inputNodes[0] XOR false
      inputNodes.push_back(new LiteralBool(false));
    } else if (gateType == OpSymb::LogCompOp::logicalAnd) {
      // inputNodes[0] AND true
      inputNodes.push_back(new LiteralBool(true));
    } else {
      throw std::runtime_error(
          "Method rewriteMultiInputGateToBinaryGatesChain currently supports 1-input gates of type logical-XOR "
          "or logical-AND only.");
    }
  }

  // vector of resulting binary gates
  std::vector<Node*> outputNodes;

  // handle first "special" gate -> takes two inputs as specified in inputNodes
  auto it = std::begin(inputNodes);
  auto recentLexp = new LogicalExpr((*it++)->castTo<AbstractExpr>(), gateType, (*it++)->castTo<AbstractExpr>());
  outputNodes.push_back(recentLexp);

  // handle all other gates -> are connected with each other
  for (auto end = std::end(inputNodes); it != end; ++it) {
    auto newLexp = new LogicalExpr(recentLexp, gateType, (*it)->castTo<AbstractExpr>());
    outputNodes.push_back(newLexp);
    recentLexp = newLexp;
  }
  return outputNodes;
}
