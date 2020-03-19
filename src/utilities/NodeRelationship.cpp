#include "NodeRelationship.h"
#include <unordered_set>
#include "GraphNode.h"

NodeRelationship::NodeRelationship(RelationshipType relationship, GraphNode *gNode)
    : rship(relationship), refToGraphNode(gNode) {}

void NodeRelationship::addChild(GraphNode *child, bool addBackreference) {
  children.push_back(child);
  if (addBackreference) child->getRelationship(rship)->addParent(refToGraphNode, false);
}

void NodeRelationship::addParent(GraphNode *parent, bool addBackreference) {
  parents.push_back(parent);
  if (addBackreference) parent->getRelationship(rship)->addChild(refToGraphNode, false);
}

GraphNode *NodeRelationship::getOnlyChild() const {
  if (children.size()==1) {
    return children.at(0);
  } else if (children.size() > 1) {
    throw std::runtime_error("Call on getOnlyChild failed because GraphNode has more than one child!");
  } else { // children.size() == 0
    throw std::runtime_error("Call on getOnlyChild failed because GraphNode has no child!");
  }
}

GraphNode *NodeRelationship::getChildAtIndex(int idx) const {
  if (idx < children.size()) {
    return children.at(idx);
  } else {
    std::stringstream errorMsg;
    errorMsg << "Cannot access index (" << idx << ") ";
    errorMsg << "of node that has only " << children.size() << " elements.";
    throw std::invalid_argument(errorMsg.str());
  }
}

void NodeRelationship::traverseAndPrintNodes(std::ostream &outputStream) const {
  // nodes that were already printed, helps to detect and bypass graph cycles
  std::unordered_set<std::string> printedNodes;
  // stack of nodes to be processed next
  std::stack<std::pair<GraphNode *, int>> q;
  q.push({std::make_pair(refToGraphNode, 0)});

  while (!q.empty()) {
    auto[curNode, indentationLevel] = q.top();
    q.pop();
    outputStream << "(" << curNode->getRelationship(rship)->getChildren().size() << ") "
                 << std::string(indentationLevel, '\t')
                 << curNode->getRefToOriginalNode()->getUniqueNodeId()
                 << std::endl;
    // continue with next While-loop iteration if this node was already printed once - avoids iterating endless if
    // there is a cycle in the graph
    if (printedNodes.count(curNode->getRefToOriginalNode()->getUniqueNodeId()) > 0) {
      if (!curNode->getRelationship(rship)->getChildren().empty()) {
        outputStream << "    " << std::string(indentationLevel, '\t')
                     << "... see above, visiting an already visited node ..." << std::endl;
      }
      continue;
    }
    printedNodes.emplace(curNode->getRefToOriginalNode()->getUniqueNodeId());
    // as we are using a stack, we need to add the children in reverse order
    for (auto it = curNode->getRelationship(rship)->getChildren().rbegin();
         it!=curNode->getRelationship(rship)->getChildren().rend(); ++it) {
      q.push(std::make_pair(*it, indentationLevel + 1));
    }
  }
}

bool NodeRelationship::areEqualGraphs(GraphNode *rootNodeOther) const {
  // nodes that were already visited, helps to detect and bypass graph cycles
  std::unordered_set<GraphNode *> visitedNodes;
  // define queues to be used to define nodes to process next
  std::stack<GraphNode *> qOne{{refToGraphNode}};
  std::stack<GraphNode *> qOther{{rootNodeOther}};

  while (!qOne.empty()) {
    auto oneCur = qOne.top();
    auto otherCur = qOther.top();
    qOne.pop();
    qOther.pop();
    // check that the number of child and parent nodes is equal
    if (oneCur->getRelationship(rship)->getChildren().size()!=otherCur->getRelationship(rship)->getChildren().size()
        || oneCur->getRelationship(rship)->getParents().size()!=otherCur->getRelationship(rship)->getParents().size()) {
      return false;
    }
    if (visitedNodes.count(oneCur) > 0) {
      continue;
    }
    visitedNodes.insert(oneCur);
    for (int i = 0; i < oneCur->getRelationship(rship)->getChildren().size(); ++i) {
      qOne.push(oneCur->getRelationship(rship)->getChildren().at(i));
      qOther.push(otherCur->getRelationship(rship)->getChildren().at(i));
    }
  }
  // ensure that qOne and qOther are empty (qOne is empty because while-loop ended)
  return qOther.empty();
}

const std::vector<GraphNode *> &NodeRelationship::getChildren() const {
  return children;
}

const std::vector<GraphNode *> &NodeRelationship::getParents() const {
  return parents;
}
