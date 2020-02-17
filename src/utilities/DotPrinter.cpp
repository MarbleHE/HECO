#include "DotPrinter.h"

DotPrinter::DotPrinter() : mdc(nullptr) {}

DotPrinter &DotPrinter::setShowMultDepth(bool show_mult_depth) {
    showMultDepth = show_mult_depth;
    if (show_mult_depth && mdc == nullptr) {
        throw std::logic_error(
                "Printing the multiplicative depth for nodes requires providing a MultiplicativeDepthCalculator instance "
                "using setMultiplicativeDepthsCalculator(...) prior calling setShowMultDepth(true).");
    }
    return *this;
}

DotPrinter &DotPrinter::setIndentationCharacter(const std::string &indentation_character) {
    indentationCharacter = indentation_character;
    return *this;
}

DotPrinter &DotPrinter::setOutputStream(std::ostream &stream) {
    outputStream = &stream;
    return *this;
}

DotPrinter &
DotPrinter::setMultiplicativeDepthsCalculator(MultiplicativeDepthCalculator &multiplicativeDepthCalculator) {
    mdc = &multiplicativeDepthCalculator;
    return *this;
}

std::string DotPrinter::getDotFormattedString(Node *n) {
    // we cannot print the node as DOT graph if it does not support the circuit mode (child/parent relationship)
    if (!n->supportsCircuitMode())
        throw std::logic_error(
                "Cannot execute 'getDotFormattedString(" + n->getUniqueNodeId() +
                ") as node is not circuit-compatible!");

    std::stringstream finalString;

    // depending on whether the graph is reversed we are interested in the parents or children
    auto vec = (n->hasReversedEdges() ? n->getParentsNonNull() : n->getChildrenNonNull());

    // vec.empty(): only print node details (e.g., operator type for Operator) for tree leaves
    finalString << dotVertex(n, this->showMultDepth, this->mdc, vec.empty())
            .buildVertexString(this->indentationCharacter);

    // only print edges if there are any edges at all
    if (vec.empty()) return finalString.str();

    // otherwise also generate string for edge and return both
    finalString << dotEdge(n, n->hasReversedEdges()).buildEdgeString(this->indentationCharacter);
    return finalString.str();
}

void DotPrinter::printAsDotFormattedGraph(Ast &ast) {
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

void DotPrinter::printAllReachableNodes(Node *pNode) {
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
