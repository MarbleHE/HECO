#include <LogicalExpr.h>
#include <set>
#include <queue>
#include "ConeRewriter.h"
#include "Return.h"
#include "Function.h"

std::vector<Node*> ConeRewriter::getReducibleCones(Node* v) {
  return getReducibleCones(v, getMultDepth(v) + 1);
}

std::vector<Node*> ConeRewriter::getReducibleCones(Node* v, int minDepth) {
  // return empty set if minDepth is reached
  if (getMultDepth(v) == minDepth) return std::vector<Node*>();

  // get critical predecessor nodes: { p ∈ pred(v) | l(p) = l(v) - d(v) }
  std::vector<Node*>* P = getCriticalPredecessors(v);

  // return v if at least one predecessor of v is non-critical and v is an AND-gate
  auto logicalExp = dynamic_cast<LogicalExpr*>(v);
  if (P->size() < 2 && logicalExp != nullptr && logicalExp->getOp().equals(OpSymb::logicalAnd)) {
    // return set consisting of start node v only
    return std::vector<Node*>{v};
  }

  // determine reducible input cones
  std::vector<std::vector<Node*>> deltaR;
  std::vector<Node*> delta;
  for (auto &p : *P) {
    std::vector<Node*> intermedResult = getReducibleCones(p);
    if (!intermedResult.empty()) deltaR.push_back(intermedResult);
  }

  // return empty set if either one of the following is true:
  // a. v is not a LogicalExpr
  // b. v is a LogicalExpr but not an AND- or XOR-gate
  // b. v is an AND-gate and deltaR is empty
  // c. v is a XOR-gate and size of deltaR does not equal size of P
  if (logicalExp == nullptr ||
      !(logicalExp->getOp().equals(OpSymb::logicalAnd) || logicalExp->getOp().equals(OpSymb::logicalXor)) ||
      (logicalExp->getOp().equals(OpSymb::logicalAnd) && deltaR.empty()) ||
      (logicalExp->getOp().equals(OpSymb::logicalXor) && deltaR.size() != P->size())) {
    return std::vector<Node*>();
  }

  if (logicalExp->getOp().equals(OpSymb::logicalAnd)) {
    // both cones must be reducible because deltaR is non-empty -> pick a random one, and assign to delta
    delta = *select_randomly(deltaR.begin(), deltaR.end());
  } else if (logicalExp->getOp().equals(OpSymb::logicalXor)) {
    // critical cones must be reducible because size of deltaR equals size of P
    // flatten vector deltaR consisting of sets generated each by getReducibleCones
    std::vector<Node*> flattenedDeltaR;
    flattenVectors(flattenedDeltaR, deltaR);
    // add all elements of flattened deltaR to delta
    addElements(delta, flattenedDeltaR);
  }

  // return 𝛅 ⋃ {v}
  delta.push_back(v);
  return delta;
}

int depthValue(Node* n) {
  if (auto lexp = dynamic_cast<LogicalExpr*>(n)) {
    return (lexp->getOp().equals(OpSymb::logicalAnd)) ? 1 : 0;
  }
  return 0;
}

int ConeRewriter::getMultDepth(Node* v) {
  if (v->getParents().empty()) {
    return 0;
  }
  int max = 0;
  for (auto &u : *getPred(v)) {
    max = std::max(getMultDepth(u) + depthValue(v), max);
  }
  return max;
}

void ConeRewriter::addElements(std::vector<Node*> &result, std::vector<Node*> newElements) {
  result.reserve(result.size() + newElements.size());
  result.insert(result.end(), newElements.begin(), newElements.end());
}

void ConeRewriter::flattenVectors(std::vector<Node*> &resultVector, std::vector<std::vector<Node*>> vectorOfVectors) {
  std::for_each(vectorOfVectors.begin(), vectorOfVectors.end(), [&](std::vector<Node*> rVec) {
    resultVector.insert(resultVector.end(), rVec.begin(), rVec.end());
  });
}

std::vector<Node*>* ConeRewriter::getCriticalPredecessors(Node* v) {
  auto result = new std::vector<Node*>();
  int criterion = getMultDepth(v) - depthValue(v);
  for (auto &p : *getPred(v)) {
    if (getMultDepth(p) == criterion) result->push_back(p);
  }
  return result;
}

std::vector<Node*>* ConeRewriter::getPred(Node* n) {
  auto result = new std::set<Node*>;
  std::queue<Node*> processQueue;
  processQueue.push(n);
  while (!processQueue.empty() && n != nullptr) {
    auto parentNodes = n->getParents();
    std::for_each(parentNodes.begin(), parentNodes.end(), [&](Node* n) {
      result->insert(n);
      processQueue.push(n);
    });
    n = processQueue.front();
    processQueue.pop();
  }
  return new std::vector<Node*>(result->begin(), result->end());
}

bool ConeRewriter::isValidCircuit(Ast &ast) {
  // At the moment only circuits consisting of a Return statement and a LogicalExp are supported
  std::set<std::string>
      acceptedNodeNames = {LogicalExpr().getNodeName(), Return().getNodeName(), Function().getNodeName()};
  std::queue<Node*> nodesToCheck;
  nodesToCheck.push(ast.getRootNode());
  while (!nodesToCheck.empty()) {
    auto curNode = nodesToCheck.front();
    nodesToCheck.pop();
    if (!acceptedNodeNames.count(curNode->getNodeName())) return false;
  }
  return true;
}

void ConeRewriter::reverseEdges(Ast &ast) {
  std::queue<Node*> nodesToCheck;
  nodesToCheck.push(ast.getRootNode());
  while (!nodesToCheck.empty()) {
    auto curNode = nodesToCheck.front();
    nodesToCheck.pop();
    std::for_each(curNode->getChildren().begin(), curNode->getChildren().end(), [&](Node* n) {
      nodesToCheck.push(n);
    });
    curNode->swapChildrenParents();
  }
}
