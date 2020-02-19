#include <sstream>
#include <queue>
#include <set>
#include <iostream>
#include "AbstractNode.h"

int AbstractNode::nodeIdCounter = 0;

std::string AbstractNode::genUniqueNodeId() {
  if (assignedNodeId==-1) {
    throw std::logic_error("Could not find any reserved ID for node. "
                           "Node constructor needs to reserve ID for node (see empty constructor).");
  }

  // build and return the node ID string
  std::stringstream ss;
  ss << getNodeName() << "_" << assignedNodeId;
  return ss.str();
}

AbstractNode::AbstractNode() {
  // save the ID reserved for this node but do not
  assignedNodeId = getAndIncrementNodeId();
}

std::string AbstractNode::getUniqueNodeId() {
  // if there is no ID defined yet, create and assign an ID
  if (uniqueNodeId.empty()) this->uniqueNodeId = genUniqueNodeId();
  // otherwise just return the previously generated ID
  return uniqueNodeId;
}

int AbstractNode::getAndIncrementNodeId() {
  return nodeIdCounter++;
}

void AbstractNode::resetNodeIdCounter() {
  AbstractNode::nodeIdCounter = 0;
}

const std::vector<AbstractNode *> &AbstractNode::getChildren() const {
  return children;
}

std::vector<AbstractNode *> AbstractNode::getChildrenNonNull() const {
  std::vector<AbstractNode *> childrenFiltered;
  std::copy_if(children.begin(), children.end(), std::back_inserter(childrenFiltered),
               [](AbstractNode *n) { return n!=nullptr; });
  return childrenFiltered;
}

std::vector<AbstractNode *> AbstractNode::getParentsNonNull() const {
  std::vector<AbstractNode *> parentsFiltered;
  std::copy_if(parents.begin(), parents.end(), std::back_inserter(parentsFiltered),
               [](AbstractNode *n) { return n!=nullptr; });
  return parentsFiltered;
}

void AbstractNode::addChild(AbstractNode *child, bool addBackReference) {
  addChildren({child}, addBackReference);
}

void AbstractNode::addChildren(const std::vector<AbstractNode *> &childrenToAdd, bool addBackReference) {
  // check whether the number of children to be added does not exceed the number of maximum supported children
  if (childrenToAdd.size() > getMaxNumberChildren() && getMaxNumberChildren()!=-1) {
    throw std::invalid_argument(
        "AbstractNode " + getUniqueNodeId() + " of type " + getNodeName() + " does not allow more than "
            + std::to_string(getMaxNumberChildren()) + " children!");
  }

  // check if circuit mode is supported by current node, otherwise addChildren will lead to unexpected behavior
  if (!this->supportsCircuitMode()) {
    throw std::logic_error(
        "Cannot use addChildren because node does not support circuit mode!");
  }

  // these actions are to be performed after a node was added to the list of children
  auto doInsertPostAction = [&](AbstractNode *childToAdd) {
    // if option 'addBackReference' is true, we add a back reference to the child as parent
    if (addBackReference) childToAdd->addParent(this);
  };

  if (getChildren().empty() || getMaxNumberChildren()
      ==
          -1) {  // if the list of children is still empty, we can simply add all nodes in one batch
    // add children to the vector's end
    children.insert(children.end(), childrenToAdd.begin(), childrenToAdd.end());
    std::for_each(children.begin(), children.end(), doInsertPostAction);
    // if this nodes accepts an infinite number of children, pre-filling the slots does not make any sense -> skip it
    if (getMaxNumberChildren()!=-1) {
      // fill remaining slots with nullptr values
      children.insert(children.end(), getMaxNumberChildren() - getChildren().size(), nullptr);
    }
  } else {  // otherwise we need to add the children one-by-one by looking for free slots
    size_t childIdx = 0;
    // add child in first empty spot
    for (auto it = getChildren().begin(); it!=getChildren().end() && childIdx < childrenToAdd.size(); ++it) {
      if (*it==nullptr) {
        auto childToAdd = childrenToAdd.at(childIdx);
        setChild(it, childToAdd);
        doInsertPostAction(childToAdd);
        childIdx++;
      }
    }
    // check if we were able to add all children, otherwise throw an exception
    if (childIdx!=childrenToAdd.size()) {
      throw std::logic_error("Cannot add one or multiple children to " + this->getUniqueNodeId()
                                 + " without overwriting an existing one. Consider removing an existing child first.");
    }
  }
}

void AbstractNode::setChild(std::vector<AbstractNode *>::const_iterator position, AbstractNode *value) {
  auto newIterator = children.insert(position, value);
  children.erase(++newIterator);
}

void AbstractNode::removeChild(AbstractNode *child) {
  auto it = std::find(children.begin(), children.end(), child);
  if (it!=children.end()) {
    // if the node supports an infinite number of children (getMaxNumberChildren() == -1), we can delete the node from
    // the children list, otherwise we just overwrite the slot with a nullptr
    if (this->getMaxNumberChildren()!=-1) {
      *it = nullptr;
    } else {
      children.erase(it);
    }
  }
  //children.erase(it);
}

void AbstractNode::removeChildBilateral(AbstractNode *child) {
  child->removeParent(this);
  this->removeChild(child);
}

void AbstractNode::isolateNode() {
  for (auto &p : getParentsNonNull()) p->removeChild(this);
  for (auto &c : getChildrenNonNull()) c->removeParent(this);
  removeChildren();
  removeParents();
}

const std::vector<AbstractNode *> &AbstractNode::getParents() const {
  return parents;
}

void AbstractNode::addParent(AbstractNode *n) {
  parents.push_back(n);
}

void AbstractNode::removeParent(AbstractNode *parent) {
  auto it = std::find(parents.begin(), parents.end(), parent);
  if (it!=parents.end()) parents.erase(it);
}

void AbstractNode::removeChildren() {
  children.clear();
}

void AbstractNode::removeParents() {
  parents.clear();
}

void AbstractNode::addParentTo(AbstractNode *parentNode, std::vector<AbstractNode *> nodesToAddParentTo) {
  std::for_each(nodesToAddParentTo.begin(), nodesToAddParentTo.end(), [&](AbstractNode *n) {
    if (n!=nullptr) n->addParent(parentNode);
  });
}

void AbstractNode::swapChildrenParents() {
  std::vector<AbstractNode *> oldParents = this->parents;
  this->parents = this->children;
  this->children = oldParents;
  // toggle the isReversed boolean
  isReversed = !isReversed;
}

void AbstractNode::accept(Visitor &v) {
  std::cout << "This shouldn't be executed!" << std::endl;
}

void to_json(json &j, const AbstractNode &n) {
  j = n.toJson();
}

json AbstractNode::toJson() const {
  return json({"type", "AbstractNode"});
}

std::string AbstractNode::toString() const {
  return this->toJson().dump();
}

std::ostream &operator<<(std::ostream &os, const std::vector<AbstractNode *> &v) {
  os << "[";
  for (int i = 0; i < v.size(); ++i) {
    os << v[i]->getUniqueNodeId();
    if (i!=v.size() - 1)
      os << ", ";
  }
  os << "]";
  return os;
}

void AbstractNode::setUniqueNodeId(const std::string &unique_node_id) {
  uniqueNodeId = unique_node_id;
}

std::vector<AbstractNode *> AbstractNode::getAncestors() {
  // use a set to avoid duplicates as there may be common ancestors between this node and any of the node's parents
  std::set<AbstractNode *> result;
  std::queue<AbstractNode *> processQueue{{this}};
  while (!processQueue.empty()) {
    auto curNode = processQueue.front();
    processQueue.pop();
    auto nextNodes = curNode->getParents();
    std::for_each(nextNodes.begin(), nextNodes.end(), [&](AbstractNode *node) {
      result.insert(node);
      processQueue.push(node);
    });
  }
  return std::vector<AbstractNode *>(result.begin(), result.end());
}

std::vector<AbstractNode *> AbstractNode::getDescendants() {
  // use a set to avoid duplicates as there may be common descendants between this node and any of the node's children
  std::set<AbstractNode *> result;
  std::queue<AbstractNode *> processQueue{{this}};
  while (!processQueue.empty()) {
    auto curNode = processQueue.front();
    processQueue.pop();
    for (auto &node : curNode->getChildrenNonNull()) {
      result.insert(node);
      processQueue.push(node);
    }
  }
  return std::vector<AbstractNode *>(result.begin(), result.end());
}

bool AbstractNode::hasParent(AbstractNode *n) {
  return std::any_of(getParents().begin(), getParents().end(), [&n](AbstractNode *p) { return (p==n); });
}

int AbstractNode::countChildrenNonNull() const {
  return std::count_if(getChildren().begin(), getChildren().end(), [](AbstractNode *n) { return n!=nullptr; });
}

int AbstractNode::getMaxNumberChildren() {
  return 0;
}

bool AbstractNode::supportsCircuitMode() {
  return false;
}

AbstractNode *AbstractNode::getChildAtIndex(int idx) const {
  return getChildAtIndex(idx, false);
}

AbstractNode *AbstractNode::getChildAtIndex(int idx, bool isEdgeDirectionAware) const {
  try {
    return (isEdgeDirectionAware && isReversed) ? parents.at(idx) : children.at(idx);
  } catch (std::out_of_range const &e) {
    return nullptr;
  }
}

AbstractNode::~AbstractNode() = default;

bool AbstractNode::hasReversedEdges() const {
  return isReversed;
}
AbstractNode *AbstractNode::cloneFlat() {
  //TODO(vianda): Implement cloneFlat
  throw std::runtime_error("Not implemented");
}
