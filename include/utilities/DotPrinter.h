#ifndef AST_OPTIMIZER_DOTPRINTER_H
#define AST_OPTIMIZER_DOTPRINTER_H

#include <string>
#include <sstream>
#include <queue>
#include <vector>
#include <deque>
#include <set>
#include <utility>
#include "Operator.h"
#include "AbstractStatement.h"
#include "Node.h"
#include "LogicalExpr.h"
#include "MultiplicativeDepthCalculator.h"

// Dot vertex represents a Node in the DOT graph language
// e.g., Return_1 [label="Return_1\n[l(v): 3, r(v): 0]" shape=oval style=filled fillcolor=white]
struct dotVertex {
 private:
  // variables with default values
  std::string identifier;
  std::string shape{"oval"};
  std::string style{"filled"};
  std::string fillColor{"white"};
  std::string details;
  std::string multDepthString;

  // a reference to the node this DotVertex represents
  Node *node{};

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
    auto lexp = dynamic_cast<LogicalExpr *>(node);
    if (lexp==nullptr) return fillColor;
    return (lexp->getOp()!=nullptr && lexp->getOp()->equals(OpSymb::logicalAnd)) ? "red" : fillColor;
  }

  std::string getShape() {
    return (dynamic_cast<AbstractStatement *>(node)!=nullptr ? "rect" : shape);
  }

  std::string getMultDepthsString() {
    return multDepthString;
  }

  void buildMultDepthsString(MultiplicativeDepthCalculator &mdc) {
    auto L = mdc.getMultDepthL(node);
    auto R = mdc.getReverseMultDepthR(node);
    std::stringstream ss;
    ss << "\\n[";
    ss << "l(v): " << std::to_string(L) << ", ";
    ss << "r(v): " + std::to_string(R);
    ss << +"]";
    multDepthString = ss.str();
  }

  std::string getStyle() {
    return style;
  }

 public:
  dotVertex(Node *node, bool showMultDepth, MultiplicativeDepthCalculator *mdc, bool showDetails) : node(node) {
    // show multiplicative depth in the tree nodes depending on parameter showMultDepth
    if (showMultDepth) buildMultDepthsString(*mdc);
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
struct dotEdge {
 private:
  std::string lhsArrow;
  std::string rhsArrow;

  static std::string buildCommaSeparatedList(std::vector<Node *> vec) {
    std::stringstream outputStr;
    for (auto ci = vec.begin(); ci!=vec.end(); ++ci) {
      outputStr << (*ci)->getUniqueNodeId();
      // omit comma after last item
      if ((ci + 1)!=vec.end()) outputStr << ", ";
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
  dotEdge(Node *n, bool isReversedEdge) {
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
    ss << " } -> { ";
    ss << getRhsArrow();
    ss << " }";
    ss << std::endl;
    return ss.str();
  }
};

/// Utilities to print the AST in DOT language as used by graphviz, see https://www.graphviz.org/doc/info/lang.html.
class DotPrinter {
 protected:
  MultiplicativeDepthCalculator *mdc;
  bool showMultDepth{false};
  std::string indentationCharacter{'\t'};
  std::ostream *outputStream{&std::cout};

 public:
  DotPrinter();

  DotPrinter &setShowMultDepth(bool show_mult_depth);

  DotPrinter &setIndentationCharacter(const std::string &indentation_character);

  DotPrinter &setOutputStream(std::ostream &stream);

  DotPrinter &setMultiplicativeDepthsCalculator(MultiplicativeDepthCalculator &multiplicativeDepthCalculator);

  std::string getDotFormattedString(Node *n);

  void printAsDotFormattedGraph(Ast &ast);

  void printAllReachableNodes(Node *pNode);
};

#endif //AST_OPTIMIZER_DOTPRINTER_H
