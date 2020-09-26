#include "ast_opt/visitor/ControlFlowGraph/NodeRelationship.h"
#include "ast_opt/visitor/ControlFlowGraph/GraphNode.h"
#include "ast_opt/ast/AbstractNode.h"

// This operator is required for calling count(...) on a container with <std::reference_wrapper<GraphNode> elements
bool operator<(const std::reference_wrapper<GraphNode> &fk, const std::reference_wrapper<GraphNode> &lk) {
  return fk.get() < lk.get();
}

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
    // get the next node, i.e., the one on the stack's top
    auto[curNode, indentationLevel] = q.top();
    q.pop();

    // extract required information of current node
    const auto numChildren = curNode.get().getRelationship(relationshipType).getChildren().size();
    auto uniqueNodeId = curNode.get().getAstNode().getUniqueNodeId();

    // write output string
    outputStream << "(" << numChildren << ") "
                 << std::string(indentationLevel, '\t')
                 << uniqueNodeId
                 << std::endl;

    // continue with next While-loop iteration if this node was already printed once (avoids infinite loop)
    if (printedNodesById.count(uniqueNodeId) > 0) {
      if (numChildren > 0) {
        outputStream << "    " << std::string(indentationLevel, '\t')
                     << "... see above, visiting an already visited node ..." << std::endl;
      }
      continue;
    }

    // remember that we visited this node to not visit its children again
    printedNodesById.emplace(curNode.get().getAstNode().getUniqueNodeId());

    // as we are using a stack, we need to add the children in reverse order
    auto currentNodeChildren = curNode.get().getRelationship(relationshipType).getChildren();
    for (auto it = currentNodeChildren.rbegin(); it!=currentNodeChildren.rend(); ++it) {
      q.push(std::make_pair(*it, indentationLevel + 1));
    }
  }
}

bool NodeRelationship::isEqualToGraph(GraphNode &rootNodeOther) const {
  // nodes that were already visited, helps to detect and bypass graph cycles
  std::unordered_set<std::reference_wrapper<GraphNode>, GraphNodeHashFunction> visitedNodes;

  // define queues to be used to define nodes to process next
  std::stack<std::reference_wrapper<GraphNode>> qOne;
  qOne.emplace(graphNode);
  std::stack<std::reference_wrapper<GraphNode>> qOther;
  qOther.emplace(rootNodeOther);

  while (!qOne.empty()) {
    // retrieve next nodes t
    GraphNode &thisCurrentNode = qOne.top().get();
    qOne.pop();
    GraphNode &otherCurrentNode = qOther.top().get();
    qOther.pop();

    // check that the number of child and parent nodes is equal
    auto thisChildren = thisCurrentNode.getRelationship(relationshipType).getChildren();
    auto otherChildren = otherCurrentNode.getRelationship(relationshipType).getChildren();
    const auto thisNumParents = thisCurrentNode.getRelationship(relationshipType).getParents().size();
    const auto otherNumParents = otherCurrentNode.getRelationship(relationshipType).getParents().size();
    if ((thisChildren.size()!=otherChildren.size()) || thisNumParents!=otherNumParents) {
      return false;
    }

    // check if we visited thisCurrentNode in the past
    if (visitedNodes.count(thisCurrentNode) > 0) {
      continue;
    }
    // otherwise remember that we visited this node
    // it is sufficient to do this by considering the current graph only
    visitedNodes.emplace(thisCurrentNode);

    // enqueue all children of thisCurrentNode and otherCurrentNode
    for (int i = 0; i < thisChildren.size(); ++i) {
      qOne.push(thisCurrentNode.getRelationship(relationshipType).getChildren().at(i));
      qOther.push(otherChildren.at(i));
    }
  }

  // ensure that qOne and qOther are empty (qOne is empty because while-loop ended)
  return qOther.empty();
}

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

inline bool operator==(std::reference_wrapper<GraphNode> const &lhs, std::reference_wrapper<GraphNode> const &rhs) {
  return lhs.get()==rhs.get();
}

std::vector<std::reference_wrapper<GraphNode>> NodeRelationship::getAllReachableNodes() const {
  // the set of nodes that we already  nodes; this is needed because our CFG can contain cycles (e.g., For loop)
  std::set<std::string> visitedNodes_uniqueNodeIds;
  std::vector<std::reference_wrapper<GraphNode>> visitedNodes;
  // the set of nodes that we did not visit yet
  std::stack<std::reference_wrapper<GraphNode>> nextNodeToVisit;
  nextNodeToVisit.emplace(this->graphNode);
  while (!nextNodeToVisit.empty()) {
    auto curNode = nextNodeToVisit.top();
    nextNodeToVisit.pop();
    // if this node was processed before: do not visit it again, otherwise we'll end up in an infinite loop
    if (visitedNodes_uniqueNodeIds.count(curNode.get().getAstNode().getUniqueNodeId()) > 0) { continue; }
    // remember that we visited this node
    visitedNodes_uniqueNodeIds.insert(curNode.get().getAstNode().getUniqueNodeId());
    visitedNodes.push_back(curNode);
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
