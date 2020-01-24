#include "Node.h"
#include <sstream>

int Node::nodeIdCounter = 0;

std::string Node::genUniqueNodeId() {
  std::stringstream ss;
  ss << getNodeName();
  ss << "_";
  ss << getAndIncrementNodeId();
  return ss.str();
}

Node::Node() {}

std::string Node::getUniqueNodeId() {
  if (uniqueNodeId.empty()) {
    std::string nodeId = genUniqueNodeId();
    this->uniqueNodeId = nodeId;
  }
  return uniqueNodeId;
}

int Node::getAndIncrementNodeId() {
  int current = Node::getNodeIdCounter();
  Node::nodeIdCounter += 1;
  return current;
}

int Node::getNodeIdCounter() {
  return nodeIdCounter;
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

void Node::addChild(Node* child) {
  children.push_back(child);
}

void Node::addChildren(std::vector<Node*> c) {
  std::for_each(c.begin(), c.end(), [&](Node* n) {
    children.push_back(n);
  });
}

void Node::removeChild(Node* child) {
  auto it = std::find(children.begin(), children.end(), child);
  if (it != children.end()) children.erase(it);
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

void Node::addParent(Node* parentNode, std::vector<Node*> nodesToAddParentTo) {
  std::for_each(nodesToAddParentTo.begin(), nodesToAddParentTo.end(), [&](Node* n) {
    n->addParent(parentNode);
  });
}

void Node::swapChildrenParents() {
  auto oldParents = this->parents;
  this->parents = std::move(this->children);
  this->children = std::move(oldParents);
}
