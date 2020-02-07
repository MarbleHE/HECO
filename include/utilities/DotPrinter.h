#ifndef AST_OPTIMIZER_INCLUDE_UTILITIES_DOTPRINTER_H_
#define AST_OPTIMIZER_INCLUDE_UTILITIES_DOTPRINTER_H_

#include <string>
#include <sstream>
#include <deque>
#include "../include/ast/Operator.h"
#include "../include/ast/AbstractStatement.h"
#include "../include/ast/Node.h"
#include "../include/ast/LogicalExpr.h"

/// Utilities to print the AST in DOT language as used by graphviz, see https://www.graphviz.org/doc/info/lang.html.
namespace DotPrinter {

// Dot vertex represents a Node in the DOT graph language
// e.g., Return_1 [label="Return_1\n[l(v): 3, r(v): 0]" shape=oval style=filled fillcolor=white]
struct DotVertex {
 private:
  // variables with default values
  std::string identifier;
  std::string shape{"oval"};
  std::string style{"filled"};
  std::string fillColor{"white"};
  std::string details;
  std::string multDepthString;

  // a reference to the node this DotVertex represents
  Node* node{};

  void buildDetailsString() {
    details = "\\n" + node->toString();
  }

  std::string getDetails() {
    return details;
  }

  std::string getIdentifier() {
    return node->getUniqueNodeId();
  }

  std::string getFillColor() {
    auto lexp = dynamic_cast<LogicalExpr*>(node);
    if (lexp == nullptr) return fillColor;
    return (lexp->getOp() != nullptr && lexp->getOp()->equals(OpSymb::logicalAnd)) ? "red" : fillColor;
  }

  std::string getShape() {
    return (dynamic_cast<AbstractStatement*>(node) != nullptr ? "rect" : shape);
  }

  std::string getMultDepthsString() {
    return multDepthString;
  }

  void buildMultDepthsString() {
    auto L = node->getMultDepthL();
    auto R = node->getReverseMultDepthR();
    multDepthString = "\\n[l(v): " + std::to_string(L) + ", r(v): " + std::to_string(R) + "]";
  }

  std::string getStyle() {
    return style;
  }

 public:
  DotVertex(Node* node, bool showMultDepth, bool showDetails) : node(node) {
    // show multiplicative depth in the tree nodes depending on parameter showMultDepth
    if (showMultDepth) buildMultDepthsString();
    // show extra information if this node is a leaf node
    if (showDetails) buildDetailsString();
  }

  std::string buildVertexString(const std::string &indentation) {
    // print node data, e.g., Return_1 [label="Return_1\n[l(v): 3, r(v): 0]" shape=oval style=filled fillcolor=white]
    std::stringstream ss;
    ss << indentation << getIdentifier() << " [";
    ss << "label=\"" << getIdentifier() << getMultDepthsString() << getDetails() << "\" ";
    ss << "shape=" << getShape() << " ";
    ss << "style=" << getStyle() << " ";
    ss << "fillcolor=" << getFillColor();
    ss << "]" << std::endl;
    return ss.str();
  }
};

/// DotEdge represents an edge between Nodes in the DOT graph language
/// e.g., { LogicalExpr_3, Operator_4, Variable_5 } -> LogicalExpr_2
struct DotEdge {
 private:
  std::string lhsArrow;
  std::string rhsArrow;

  static std::string buildCommaSeparatedList(std::vector<Node*> vec) {
    std::stringstream outputStr;
    for (auto ci = vec.begin(); ci != vec.end(); ++ci) {
      outputStr << (*ci)->getUniqueNodeId();
      // omit comma after last item
      if ((ci + 1) != vec.end()) outputStr << ", ";
    }
    return outputStr.str();
  }

  std::string getLhsArrow() {
    return lhsArrow;
  }

  std::string getRhsArrow() {
    return rhsArrow;
  }

 public:
  DotEdge(Node* n, bool isReversedEdge) {
    if (isReversedEdge) {
      lhsArrow = buildCommaSeparatedList(n->getParentsNonNull());
      rhsArrow = n->getUniqueNodeId();
    } else {
      lhsArrow = n->getUniqueNodeId();
      rhsArrow = buildCommaSeparatedList(n->getChildrenNonNull());
    }
  }

  std::string buildEdgeString(const std::string &indentation) {
    std::stringstream ss;
    ss << indentation;
    ss << "{ ";
    ss << getLhsArrow();
    ss << " } -> {";
    ss << getRhsArrow();
    ss << " }";
    ss << std::endl;
    return ss.str();
  }
};

static std::string getDotFormattedString(Node* n, const std::string &indentation, bool showMultDepth = false) {
  std::stringstream finalString;

  // depending on whether the graph is reversed we are interested in the parents or children
  auto vec = (n->hasReversedEdges() ? n->getParentsNonNull() : n->getChildrenNonNull());

  // vec.empty(): only print node details (e.g., operator type for Operator) for tree leaves
  finalString << DotVertex(n, showMultDepth, vec.empty()).buildVertexString(indentation);

  // only print edges if there are any edges at all
  if (vec.empty()) return finalString.str();

  // otherwise also generate string for edge and return both
  finalString << DotEdge(n, n->hasReversedEdges()).buildEdgeString(indentation);
  return finalString.str();
}

void printAsDotFormattedGraph(Ast &ast) {
  std::stringstream ss;
  ss << "digraph D {" << std::endl;
  std::deque<std::pair<Node*, int>> q;
  q.emplace_back(ast.getRootNode(), 1);
  while (!q.empty()) {
    auto curNode = q.front().first;
    auto il = q.front().second;
    q.pop_front();
    ss << getDotFormattedString(curNode, "\t", true);
    auto nodes = (ast.isReversed()) ? curNode->getParentsNonNull() : curNode->getChildrenNonNull();
    for (auto &n : nodes) q.emplace_front(n, il + 1);
  }
  ss << "}" << std::endl;
  std::cout << ss.str();
}

void printAllReachableNodes(Node* pNode) {
  std::set<Node*> printedNodes;
  std::queue<Node*> q{{pNode}};
  while (!q.empty()) {
    auto curNode = q.front();
    q.pop();
    if (printedNodes.count(curNode) == 0) {
      std::cout << DotPrinter::getDotFormattedString(curNode, "\t", false);
      for (auto &c : curNode->getChildrenNonNull()) q.push(c);
      for (auto &p : curNode->getParentsNonNull()) q.push(p);
      printedNodes.insert(curNode);
    }
  }
}

}

#endif //AST_OPTIMIZER_INCLUDE_UTILITIES_DOTPRINTER_H_
