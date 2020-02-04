#include <sstream>
#include <iostream>
#include <set>
#include <queue>
#include "../include/optimizer/ConeRewriter.h"
#include "AbstractExpr.h"
#include "Node.h"
#include "LogicalExpr.h"

int Node::nodeIdCounter = 0;

std::string Node::genUniqueNodeId() {
    std::stringstream ss;
    ss << getNodeName();
    ss << "_";
    ss << getAndIncrementNodeId();
    return ss.str();
}

Node::Node() = default;

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

const std::vector<Node *> &Node::getChildren() const {
    return children;
}

void Node::addChild(Node *child) {
    children.push_back(child);
}

void Node::addChildren(std::vector<Node *> c) {
    std::for_each(c.begin(), c.end(), [&](Node *n) {
        children.push_back(n);
    });
}

void Node::removeChild(Node *child) {
    auto it = std::find(children.begin(), children.end(), child);
    if (it != children.end()) children.erase(it);
}

const std::vector<Node *> &Node::getParents() const {
    return parents;
}

void Node::addParent(Node *n) {
    parents.push_back(n);
}

void Node::removeParent(Node *parent) {
    auto it = std::find(parents.begin(), parents.end(), parent);
    if (it != parents.end()) parents.erase(it);
}

void Node::removeChildren() {
    children.clear();
}

void Node::removeParents() {
    parents.clear();
}

void Node::addParent(Node *parentNode, std::vector<Node *> nodesToAddParentTo) {
    std::for_each(nodesToAddParentTo.begin(), nodesToAddParentTo.end(), [&](Node *n) {
        n->addParent(parentNode);
    });
}

void Node::swapChildrenParents() {
    std::vector<Node *> oldParents = this->parents;
    this->parents = this->children;
    this->children = oldParents;
}

Literal *Node::evaluate(Ast &ast) {
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

std::string Node::getDotFormattedString(bool isReversed, const std::string &indentation, bool showMultDepth) {
    // depending on whether the graph is reversed we are interested in the parents or children
    auto vec = (isReversed) ? this->getParents() : this->getChildren();

    // -------------------
    // print node data
    // e.g., 	Return_1 [label="Return_1\n[l(v): 3, r(v): 0]" shape=oval style=filled fillcolor=white]
    // -------------------
    std::stringstream ss;
    ss << indentation << this->getUniqueNodeId() << " [";

    // only print node details (e.g., operator type for Operator) for tree leaves
    std::string nodeDetails;
    if (vec.empty()) {
        nodeDetails = "\\n" + this->toString();
    }

    // show multiplicative depth in the tree nodes depending on parameter showMultDepth
    std::string multDepth;
    if (showMultDepth) {
        auto L = getMultDepthL();
        auto R = getReverseMultDepthR();
        multDepth = "\\n[l(v): " + std::to_string(L) + ", r(v): " + std::to_string(R) + "]";
    }

    // construct the string of node details
    ss << "label=\"" << this->getUniqueNodeId() << multDepth << nodeDetails << "\" ";
    std::string shape = (dynamic_cast<AbstractExpr *>(this)) ? "rect" : "oval";
    std::string fillColor("white");
    if (auto lexp = dynamic_cast<LogicalExpr *>(this)) {
        if (lexp->getOp().equals(OpSymb::logicalAnd)) {
            fillColor = "red";
        }
    }
    ss << "shape=" << shape << " ";
    ss << "style=filled fillcolor=" << fillColor;
    ss << "]" << std::endl;

    // only print edges if there are any edges at all
    if (vec.empty()) goto end;

    // -------------------
    // print edges
    // e.g., { LogicalExpr_3, Operator_4, Variable_5 } -> LogicalExpr_2
    // -------------------

    // if AST is reversed -> consider children
    if (isReversed) {
        // build a string like '{child1, child2, ..., childN} ->
        ss << indentation << "{ ";
        for (auto ci = vec.begin(); ci != vec.end(); ++ci) {
            ss << (*ci)->getUniqueNodeId();
            if ((ci + 1) != vec.end()) ss << ", ";
        }
        ss << " } -> " << this->getUniqueNodeId();
    } else {  // if AST is not reversed --> consider parents
        // build a string like 'nodeName -> {parent1, parent2, ..., parentN}'
        ss << indentation << this->getUniqueNodeId();
        ss << " -> {";
        for (auto ci = vec.begin(); ci != vec.end(); ++ci) {
            ss << (*ci)->getUniqueNodeId();
            if ((ci + 1) != vec.end()) ss << ", ";
        }
        ss << "}";
    }
    ss << std::endl;

    end:
    return ss.str();
}

std::ostream &operator<<(std::ostream &os, const std::vector<Node *> &v) {
    os << "[";
    for (int i = 0; i < v.size(); ++i) {
        os << v[i]->getUniqueNodeId();
        if (i != v.size() - 1)
            os << ", ";
    }
    os << "]";
    return os;
}

Node *Node::clone() {
    throw std::logic_error("ERROR: clone() not implemented for node of type " + getNodeName());
}

Node *Node::getUnderlyingNode() const {
    return underlyingNode;
}

void Node::setUnderlyingNode(Node *uNode) {
    underlyingNode = uNode;
}

void Node::setUniqueNodeId(const std::string &unique_node_id) {
    uniqueNodeId = unique_node_id;
}

void Node::printAncestorsDescendants(Node *n) {
    return Node::printAncestorsDescendants(std::vector<Node *>({n}));
}

void Node::printAncestorsDescendants(std::vector<Node *> nodes) {
    std::set<Node *> printedNodes;
    while (!nodes.empty()) {
        auto curNode = nodes.back();
        nodes.pop_back();
        if (printedNodes.count(curNode) == 0) {
            std::cout << curNode->getUniqueNodeId() << ":" << std::endl;
            std::cout << "  children: " << curNode->getChildren() << std::endl;
            std::cout << "  parents: " << curNode->getParents() << std::endl;
            printedNodes.emplace(curNode);
            std::for_each(curNode->getParents().begin(), curNode->getParents().end(),
                          [&](Node *n) { nodes.push_back(n); });
            std::for_each(curNode->getChildren().begin(), curNode->getChildren().end(),
                          [&](Node *n) { nodes.push_back(n); });
        }
    }
}

const std::vector<Node *> &Node::getPred() const {
    return getParents();
}

const std::vector<Node *> &Node::getSucc() const {
    return getChildren();
}

std::vector<Node *> Node::getAnc() {
    // use a set to avoid duplicates as there may be common ancestors between this node and any of the node's parents
    std::set<Node *> result;
    std::queue<Node *> processQueue{{this}};
    while (!processQueue.empty()) {
        auto curNode = processQueue.front();
        processQueue.pop();
        auto nextNodes = curNode->getParents();
        std::for_each(nextNodes.begin(), nextNodes.end(), [&](Node *node) {
            result.insert(node);
            processQueue.push(node);
        });
    }
    return std::vector<Node *>(result.begin(), result.end());
}

int Node::getMultDepthL() {
    // |pred(v)| = 0 <=> v does not have any parent node
    if (this->getPred().empty()) return 0;

    // otherwise return max_{u ∈ pred(v)} l(u) + d(v)
    int max = 0;
    for (auto &u : this->getPred()) {
        max = std::max(u->getMultDepthL() + this->depthValue(), max);
    }
    return max;
}


int Node::getReverseMultDepthR() {
    if (this->getChildren().empty()) return 0;
    int max = 0;
    for (auto &u : this->getSucc()) {
        max = std::max(u->getReverseMultDepthR() + u->depthValue(), max);
    }
    return max;
}


int Node::depthValue() {
    if (auto lexp = dynamic_cast<LogicalExpr *>(this)) {
        return lexp->getOp().equals(OpSymb::logicalAnd) ? 1 : 0;
    }
    return 0;
}

void Node::addParents(std::vector<Node *> c) {
    std::for_each(c.begin(), c.end(), [&](Node *n) {
        parents.push_back(n);
    });
}

void Node::ensureNonReversedEdge() {
    if (this->hasReversedEdge)
        this->swapChildrenParents();
    this->hasReversedEdge = false;
}

void Node::ensureReversedEdge() {
    if (!this->hasReversedEdge)
        this->swapChildrenParents();
    this->hasReversedEdge = true;
}

