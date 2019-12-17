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
