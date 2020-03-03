#include "DotPrinter.h"
#include "VarDecl.h"
#include "VarAssignm.h"

DotPrinter::DotPrinter() : mdc(nullptr) {}

DotPrinter &DotPrinter::setShowMultDepth(bool show_mult_depth) {
  showMultDepth = show_mult_depth;
  if (show_mult_depth && mdc==nullptr) {
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

std::string DotPrinter::getDotFormattedString(AbstractNode *n) {
  // we cannot print the node as DOT graph if it does not support the circuit mode (child/parent relationship)
  if (!n->supportsCircuitMode())
    throw std::logic_error(
        "Cannot execute 'getDotFormattedString(" + n->getUniqueNodeId() + ") as node is not circuit-compatible!");

  std::stringstream finalString;

  // depending on whether the graph is reversed we are interested in the parents or children
  auto vec = (n->hasReversedEdges() ? n->getParentsNonNull() : n->getChildrenNonNull());

  // define criteria when to print node details
  auto printNodeDetailsCriterion = (vec.empty()    // if node is a tree leaf
      || dynamic_cast<VarDecl *>(n)!=nullptr       // if node is a VarDecl (needed for the variable identifier)
      || dynamic_cast<VarAssignm *>(n)!=nullptr);  // if node is a VarAssignm (needed for the variable identifier)
  finalString << DotVertex(n, this->showMultDepth, this->mdc, printNodeDetailsCriterion)
      .buildVertexString(this->indentationCharacter);

  // only print edges if there are any edges at all
  if (vec.empty()) return finalString.str();

  // otherwise also generate string for edge and return both
  finalString << DotEdge(n, n->hasReversedEdges()).buildEdgeString(this->indentationCharacter);
  return finalString.str();
}

void DotPrinter::printAsDotFormattedGraph(Ast &ast) {
  *outputStream << "digraph D {" << std::endl;
  std::deque<std::pair<AbstractNode *, int>> q;
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

void DotPrinter::printAllReachableNodes(AbstractNode *pNode) {
  std::set<AbstractNode *> printedNodes;
  std::queue<AbstractNode *> q{{pNode}};
  while (!q.empty()) {
    auto curNode = q.front();
    q.pop();
    if (printedNodes.count(curNode)==0) {
      *outputStream << DotPrinter::getDotFormattedString(curNode);
      for (auto &c : curNode->getChildrenNonNull()) { q.push(c); }
      for (auto &p : curNode->getParentsNonNull()) { q.push(p); }
      printedNodes.insert(curNode);
    }
  }
}
