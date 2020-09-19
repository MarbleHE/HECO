
#include "ast_opt/visitor/ControlFlowGraph/NodeRelationship.h"
#include "ast_opt/visitor/ControlFlowGraph/GraphNode.h"

NodeRelationship::NodeRelationship(RelationshipType relationshipType, GraphNode &graphNode)
    : relationshipType(relationshipType), graphNode(graphNode) {

}

void NodeRelationship::addChild(GraphNode &child, bool addBackreference) {
  children.emplace_back(child);
  if (addBackreference) {
    child.getRelationship(relationshipType).addParent(graphNode, false);
  }
}

void NodeRelationship::addParent(GraphNode &parent, bool addBackreference) {
  parents.emplace_back(parent);
  if (addBackreference) {
    parent.getRelationship(relationshipType).addChild(graphNode, false);
  }
}

GraphNode &NodeRelationship::getOnlyChild() {
  // removes const from result of const counterpart, see https://stackoverflow.com/a/856839/3017719
  return const_cast<GraphNode &>(const_cast<const NodeRelationship *>(this)->getOnlyChild());
}

const GraphNode &NodeRelationship::getOnlyChild() const {
  if (children.size()==1) {
    return children.at(0);
  } else if (children.size() > 1) {
    throw std::runtime_error("Call on getOnlyChild failed because GraphNode has more than one child!");
  } else { // children.size() == 0
    throw std::runtime_error("Call on getOnlyChild failed because GraphNode has no child!");
  }
}

GraphNode &NodeRelationship::getChildAtIndex(int index) {
  // removes const from result of const counterpart, see https://stackoverflow.com/a/856839/3017719
  return const_cast<GraphNode &>(const_cast<const NodeRelationship *>(this)->getChildAtIndex(index));
}

const GraphNode &NodeRelationship::getChildAtIndex(int index) const {
  if (index < children.size()) {
    return children.at(index);
  } else {
    std::stringstream errorMsg;
    errorMsg << "Cannot access index (" << index << ") " << "of node that has only " << children.size() << " elements.";
    throw std::invalid_argument(errorMsg.str());
  }
}

void NodeRelationship::printNodes(std::ostream &outputStream) const {
  // nodes that were already printed (helps to detect and break out of graph cycles)
  std::unordered_set<std::string> printedNodesById;

  // stack of nodes to be processed next
  std::stack<std::pair<std::reference_wrapper<GraphNode>, int>> q;
  q.emplace(graphNode, 0);

  // as long as there are still unprocessed nodes
  while (!q.empty()) {
    // 
    auto[curNode, indentationLevel] = q.top();
    q.pop();

    //

    const auto numChildren = curNode.get().getRelationship(relationshipType).getChildren().size();
    auto uniqueNodeId = curNode.get().getAstNode().getUniqueNodeId();

    outputStream << "(" << numChildren << ") "
                 << std::string(indentationLevel, '\t')
                 << uniqueNodeId
                 << std::endl;
    // continue with next While-loop iteration if this node was already printed once (avoids infinite loop)
    if (printedNodesById.count(curNode.get().getAstNode().getUniqueNodeId()) > 0) {
      if (numChildren > 0) {
        outputStream << "    " << std::string(indentationLevel, '\t')
                     << "... see above, visiting an already visited node ..." << std::endl;
      }
      continue;
    }
    printedNodesById.emplace(curNode.get().getAstNode().getUniqueNodeId());

    // as we are using a stack, we need to add the children in reverse order
    auto currentNodeChildren = curNode.get().getRelationship(relationshipType).getChildren();
    for (auto it = currentNodeChildren.rbegin(); it!=currentNodeChildren.rend(); ++it) {
      q.push(std::make_pair(*it, indentationLevel + 1));
    }
  }
}

//bool NodeRelationship::areEqualGraphs(GraphNode *rootNodeOther) const {
//  // nodes that were already visited, helps to detect and bypass graph cycles
//  std::unordered_set<std::reference_wrapper<GraphNode &>> visitedNodes;
//  // define queues to be used to define nodes to process next
//  std::stack<GraphNode &> qOne;
//  qOne.em
//
//  {{ graphNode }};
//  std::stack<GraphNode &> qOther{{rootNodeOther}};
//
//  while (!qOne.empty()) {
//    auto oneCur = qOne.top();
//    auto otherCur = qOther.top();
//    qOne.pop();
//    qOther.pop();
//    // check that the number of child and parent nodes is equal
//    if (oneCur->getRelationship(relationshipType)->getChildren().size()
//        !=otherCur->getRelationship(relationshipType)->getChildren().size()
//        || oneCur->getRelationship(relationshipType)->getParents().size()
//            !=otherCur->getRelationship(relationshipType)->getParents().size()) {
//      return false;
//    }
//    if (visitedNodes.count(oneCur) > 0) {
//      continue;
//    }
//    visitedNodes.insert(oneCur);
//    for (int i = 0; i < oneCur->getRelationship(relationshipType)->getChildren().size(); ++i) {
//      qOne.push(oneCur->getRelationship(relationshipType)->getChildren().at(i));
//      qOther.push(otherCur->getRelationship(relationshipType)->getChildren().at(i));
//    }
//  }
//  // ensure that qOne and qOther are empty (qOne is empty because while-loop ended)
//  return qOther.empty();
//}

std::vector<std::reference_wrapper<GraphNode>> NodeRelationship::getChildren() {
  return children;
}

std::vector<std::reference_wrapper<const GraphNode>> NodeRelationship::getChildren() const {
  auto result = std::vector<std::reference_wrapper<const GraphNode>>(children.begin(), children.end());
  return result;
}

std::vector<std::reference_wrapper<GraphNode>> NodeRelationship::getParents() {
  return parents;
}

std::vector<std::reference_wrapper<const GraphNode>> NodeRelationship::getParents() const {
  std::vector<std::reference_wrapper<const GraphNode>> result(parents.begin(), parents.end());
  return result;
}

// This operator is required for calling count(...) on std::set<std::reference_wrapper<GraphNode>>
bool operator<(const std::reference_wrapper<GraphNode> &fk, const std::reference_wrapper<GraphNode> &lk) {
  return fk.get() < lk.get();
}

std::set<std::reference_wrapper<GraphNode>> NodeRelationship::getAllReachableNodes() const {
  // the set of nodes that we already  nodes; this is needed because our CFG can contain cycles (e.g., For loop)
  std::set<std::reference_wrapper<GraphNode>> visitedNodes;
  // the set of nodes that we did not visit yet
  std::stack<std::reference_wrapper<GraphNode>> nextNodeToVisit;
  nextNodeToVisit.emplace(this->graphNode);
  while (!nextNodeToVisit.empty()) {
    auto curNode = nextNodeToVisit.top();
    nextNodeToVisit.pop();
    // if this node was processed before: do not visit it again, otherwise we'll end up in an infinite loop
    if (visitedNodes.count(curNode) > 0) { continue; }
    // remember that we visited this node
    visitedNodes.insert(curNode);
    // enqueue children of current node (in reverse order to perform BFS from lhs to rhs)
    auto curNodeChildren = curNode.get().getControlFlowGraph().getChildren();
    for (auto it = curNodeChildren.rbegin(); it!=curNodeChildren.rend(); ++it) { nextNodeToVisit.push(*it); }
  }
  return visitedNodes;
}

bool NodeRelationship::hasChild(GraphNode &targetNode) {
  return std::any_of(children.begin(),
                     children.end(),
                     [&targetNode](GraphNode &node) { return (&node)==(&targetNode); });
}

bool NodeRelationship::hasParent(GraphNode &targetNode) {
  return std::any_of(parents.begin(),
                     parents.end(),
                     [&targetNode](GraphNode &node) { return (&node)==(&targetNode); });
}
