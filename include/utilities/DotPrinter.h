#ifndef AST_OPTIMIZER_INCLUDE_UTILITIES_DOTPRINTER_H_
#define AST_OPTIMIZER_INCLUDE_UTILITIES_DOTPRINTER_H_

#include <string>
#include <sstream>
#include <queue>
#include <vector>
#include <deque>
#include <set>
#include <utility>
#include <regex>
#include "Operator.h"
#include "AbstractStatement.h"
#include "AbstractNode.h"
#include "LogicalExpr.h"
#include "MultiplicativeDepthCalculator.h"

// Dot vertex represents an AbstractNode in the DOT graph language
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
  AbstractNode *node{};

  void buildDetailsString() {
    std::string str = node->toString(false);
    // extract attributes using regex from node's toString representation
    // regex basically matches everything in parenthesis '(...') where the parenthesis start '('  is preceded by a blank
    // and ends with a ')' or '):'
    std::regex r(R"(\s\((.+)\):?)");
    std::smatch m;
    std::regex_search(str, m, r);
    if (m.size()==2) {
      details = "\\n" + std::string(m[1]); // use first capture group of match
    } else {
      details = "";
    }
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
    return (lexp->getOperator()!=nullptr && lexp->getOperator()->equals(LogCompOp::LOGICAL_AND)) ? "red" : fillColor;
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
  DotVertex(AbstractNode *node, bool showMultDepth, MultiplicativeDepthCalculator *mdc, bool showDetails) : node(node) {
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
struct DotEdge {
 private:
  std::string lhsArrow;
  std::string rhsArrow;

  static std::string buildCommaSeparatedList(std::vector<AbstractNode *> vec) {
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
  DotEdge(AbstractNode *n, bool isReversedEdge) {
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

  std::string getDotFormattedString(AbstractNode *n);

  void printAsDotFormattedGraph(Ast &ast);

  void printAllReachableNodes(AbstractNode *pNode);
};

#endif //AST_OPTIMIZER_INCLUDE_UTILITIES_DOTPRINTER_H_
