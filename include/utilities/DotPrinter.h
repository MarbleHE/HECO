#ifndef AST_OPTIMIZER_INCLUDE_UTILITIES_DOTPRINTER_H_
#define AST_OPTIMIZER_INCLUDE_UTILITIES_DOTPRINTER_H_

#include <string>
#include <sstream>
#include <queue>
#include "Operator.h"
#include "AbstractStatement.h"
#include "Node.h"
#include "LogicalExpr.h"
#include "MultiplicativeDepthCalculator.h"
#include <vector>
#include <deque>
#include <set>
#include <utility>

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
        if (lexp == nullptr) return fillColor;
        return (lexp->getOp() != nullptr && lexp->getOp()->equals(OpSymb::logicalAnd)) ? "red" : fillColor;
    }

    std::string getShape() {
        return (dynamic_cast<AbstractStatement *>(node) != nullptr ? "rect" : shape);
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
    DotVertex(Node *node, bool showMultDepth, MultiplicativeDepthCalculator *mdc, bool showDetails) : node(node) {
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

    static std::string buildCommaSeparatedList(std::vector<Node *> vec) {
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
    DotEdge(Node *n, bool isReversedEdge) {
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
    DotPrinter() : mdc(nullptr) {}

    DotPrinter &setShowMultDepth(bool show_mult_depth) {
        showMultDepth = show_mult_depth;
        if (show_mult_depth && mdc == nullptr) {
            throw std::logic_error(
                    "Printing the multiplicative depth for nodes requires providing a MultiplicativeDepthCalculator instance "
                    "using setMultiplicativeDepthsCalculator(...) prior calling setShowMultDepth(true).");
        }
        return *this;
    }

    DotPrinter &setIndentationCharacter(const std::string &indentation_character) {
        indentationCharacter = indentation_character;
        return *this;
    }

    DotPrinter &setOutputStream(std::ostream &stream) {
        outputStream = &stream;
        return *this;
    }

    DotPrinter &setMultiplicativeDepthsCalculator(MultiplicativeDepthCalculator &multiplicativeDepthCalculator) {
        mdc = &multiplicativeDepthCalculator;
        return *this;
    }

    std::string getDotFormattedString(Node *n) {
        // we cannot print the node as DOT graph if it does not support the circuit mode (child/parent relationship)
        if (!n->supportsCircuitMode())
            throw std::logic_error(
                    "Cannot execute 'getDotFormattedString(" + n->getUniqueNodeId() +
                    ") as node is not circuit-compatible!");

        std::stringstream finalString;

        // depending on whether the graph is reversed we are interested in the parents or children
        auto vec = (n->hasReversedEdges() ? n->getParentsNonNull() : n->getChildrenNonNull());

        // vec.empty(): only print node details (e.g., operator type for Operator) for tree leaves
        finalString << DotVertex(n, this->showMultDepth, this->mdc, vec.empty())
                .buildVertexString(this->indentationCharacter);

        // only print edges if there are any edges at all
        if (vec.empty()) return finalString.str();

        // otherwise also generate string for edge and return both
        finalString << DotEdge(n, n->hasReversedEdges()).buildEdgeString(this->indentationCharacter);
        return finalString.str();
    }

    void printAsDotFormattedGraph(Ast &ast) {
        *outputStream << "digraph D {" << std::endl;
        std::deque<std::pair<Node *, int>> q;
        q.emplace_back(ast.getRootNode(), 1);
        while (!q.empty()) {
            auto curNode = q.front().first;
            auto il = q.front().second;
            q.pop_front();
            *outputStream << getDotFormattedString(curNode);
            auto nodes = (ast.isReversed()) ? curNode->getParentsNonNull() : curNode->getChildrenNonNull();
            for (auto &n : nodes) q.emplace_front(n, il + 1);
        }
        *outputStream << "}" << std::endl;
    }

    void printAllReachableNodes(Node *pNode) {
        std::set<Node *> printedNodes;
        std::queue<Node *> q{{pNode}};
        while (!q.empty()) {
            auto curNode = q.front();
            q.pop();
            if (printedNodes.count(curNode) == 0) {
                *outputStream << DotPrinter::getDotFormattedString(curNode);
                for (auto &c : curNode->getChildrenNonNull()) { q.push(c); }
                for (auto &p : curNode->getParentsNonNull()) { q.push(p); }
                printedNodes.insert(curNode);
            }
        }
    }
};

#endif //AST_OPTIMIZER_INCLUDE_UTILITIES_DOTPRINTER_H_
