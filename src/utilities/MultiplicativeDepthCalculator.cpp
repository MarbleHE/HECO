#include "AbstractNode.h"
#include "Operator.h"
#include "MultiplicativeDepthCalculator.h"

#include <utility>
#include "LogicalExpr.h"

int MultiplicativeDepthCalculator::depthValue(AbstractNode *n) {
  if (auto lexp = dynamic_cast<LogicalExpr *>(n)) {
    // the multiplicative depth considers logical AND nodes only
    return (lexp->getOp()!=nullptr && lexp->getOp()->equals(LogCompOp::logicalAnd));
  }
  return 0;
}

int MultiplicativeDepthCalculator::getInitialDepthOrNull(const std::string &nodeId) {
  try {
    return initialMultiplicativeDepths.at(nodeId);
  } catch (std::out_of_range &exception) {
    return 0;
  }
}

int MultiplicativeDepthCalculator::getMultDepthL(AbstractNode *n) {
  // check if we have calculated the multiplicative depth previously
  if (!multiplicativeDepths.empty()) {
    auto it = multiplicativeDepths.find(n->getUniqueNodeId());
    if (it!=multiplicativeDepths.end())
      return it->second;
  }

  // we need to be aware whether this node (and its whole AST, hopefully) is reversed or not
  auto nextNodesToConsider = n->isReversed ? n->getParentsNonNull() : n->getChildrenNonNull();

  // we need to compute the multiplicative depth
  // trivial case: v is a leaf node, i.e., does not have any parent node
  // |pred(v)| = 0 => multiplicative depth = 0
  if (nextNodesToConsider.empty()) {
    multiplicativeDepths[n->getUniqueNodeId()] = 0 + getInitialDepthOrNull(n->getUniqueNodeId());
    return 0;
  }

  // otherwise compute max_{u ∈ pred(v)} l(u) + d(v)
  int max = 0;
  for (auto &u : nextNodesToConsider) {
    int uDepth;
    // compute the multiplicative depth of parent u
    uDepth = getMultDepthL(u);
    // store the computed depth
    multiplicativeDepths[u->getUniqueNodeId()] = uDepth + getInitialDepthOrNull(u->getUniqueNodeId());
    max = std::max(uDepth + depthValue(n), max);
  }

  return max;
}

int MultiplicativeDepthCalculator::getReverseMultDepthR(AbstractNode *n) {
  // check if we have calculated the reverse multiplicative depth previously
  if (!multiplicativeDepthsReversed.empty()) {
    auto it = multiplicativeDepthsReversed.find(n->getUniqueNodeId());
    if (it!=multiplicativeDepthsReversed.end())
      return it->second;
  }

  // we need to be aware whether this node (and its whole AST, hopefully) is reversed or not
  auto nextNodesToConsider = n->isReversed ? n->getChildrenNonNull() : n->getParentsNonNull();

  // we need to compute the reverse multiplicative depth
  if (nextNodesToConsider.empty()) {
    multiplicativeDepthsReversed[n->getUniqueNodeId()] = 0;
    return 0;
  }

  // otherwise compute the reverse depth
  int max = 0;
  for (auto &u : nextNodesToConsider) {
    int uDepthR;
    // compute the multiplicative depth of parent u
    uDepthR = getReverseMultDepthR(u);
    // store the computed depth
    multiplicativeDepthsReversed[u->getUniqueNodeId()] = uDepthR;
    max = std::max(uDepthR + depthValue(u), max);
  }

  return max;
}

void MultiplicativeDepthCalculator::precomputeMultDepths(Ast &ast) {
  // precompute the AST's multiplicative depth and reverse multiplicative depth
  multiplicativeDepths.clear();
  multiplicativeDepthsReversed.clear();
  for (auto &node : ast.getAllNodes()) {
    getMultDepthL(node);
    getReverseMultDepthR(node);
  }

  // determine the AST's maximum multiplicative depth
  maximumMultiplicativeDepth = std::max_element(
      multiplicativeDepths.begin(), multiplicativeDepths.end(),
      [](const std::pair<const std::basic_string<char>, int> &a,
         const std::pair<const std::basic_string<char>, int> &b) {
        return a.second < b.second;
      })->second;
}

int MultiplicativeDepthCalculator::getMaximumMultiplicativeDepth() {
  return maximumMultiplicativeDepth;
}

MultiplicativeDepthCalculator::MultiplicativeDepthCalculator(Ast &ast) {
  precomputeMultDepths(ast);
}

MultiplicativeDepthCalculator::MultiplicativeDepthCalculator(Ast &ast,
                                                             std::unordered_map<std::string, int> initialDepths)
    : initialMultiplicativeDepths(std::move(initialDepths)) {
  precomputeMultDepths(ast);
}


