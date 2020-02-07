#include "ConeRewriter.h"
#include "LogicalExpr.h"
#include <set>
#include <queue>
#include "Return.h"
#include "Function.h"
#include "../../include/utilities/DotPrinter.h"


// ------------------------------------–
// Algorithms presented in the paper
// ------------------------------------–

Ast &ConeRewriter::applyConeRewriting() {
  // Ensure that we have a circuit of supported nodes only
  if (!ast.isValidCircuit()) {
    throw std::invalid_argument(
        "Cannot perform Cone Rewriting on AST because AST consists of Node types that are not circuit-compatible. "
        "These nodes do not define the required child/parent relationship of class Node.");
  }

  // We need to reverse the tree's edges to be conform to the paper
  if (!ast.isReversed()) {
    ast.reverseEdges();
    precomputeMultDepths();
  }
  return applyMinMultDepthHeuristic();
}

// TODO this method has not been tested yet as rewriteCones(...) is not fully working yet
Ast &ConeRewriter::applyMinMultDepthHeuristic() {
  // TODO create a deep copy of the input circuit:  Ast* optimizedCkt = new Ast(ast);
  Ast* optimizedCkt = new Ast();

  // compute reducible cones set
  std::vector<Node*> deltaMin = ConeRewriter::computeReducibleCones();

  // repeat rewriting as long as there are any reducible cones
  // optimizedCkt will contain all rewritten cones
  while (!deltaMin.empty()) {
    // rewrite all cones from deltaMin in the input circuit
    // we need to take the input circuit because the node refs in the cones refer to nodes in the input circuit
    rewriteCones(ast, deltaMin);

    // if depth of cone-rewritten circuit is smaller -> this is our new optimized circuit
    if (getMaxMultDepth(ast) < getMaxMultDepth(*optimizedCkt)) optimizedCkt = &ast;

    // recompute reducible cones
    deltaMin = ConeRewriter::computeReducibleCones();
  }

  // return multiplicative depth-optimized circuit
  return *optimizedCkt;
}

std::vector<Node*> ConeRewriter::computeReducibleCones() {
  // delta: set of all reducible cones
  std::vector<Node*> delta = getReducibleCones();
  std::cout << "  delta: " << std::endl << "  " << delta << std::endl;

  // cAndCkt: C^{AND} circuit consisting of critical AND-nodes that are connected if there is a multiplicative depth-2
  // path in between two of those nodes in the initial circuit
  // each cone δ ∈ Δ has an unique terminal AND node in C^{AND}
  std::vector<Node*> cAndCkt = ConeRewriter::getAndCriticalCircuit(delta);
  std::cout << "  cAndCkt:" << std::endl << "  " << cAndCkt << std::endl;

  // deltaMin: the minimum set of reducible cones
  std::vector<Node*> deltaMin = ConeRewriter::selectCones(cAndCkt);
  std::cout << "  deltaMin: " << std::endl << "  " << deltaMin << std::endl;
  return deltaMin;
}

std::vector<Node*> ConeRewriter::getReducibleCones() {
  Node* startNode = nullptr;
  auto rootNode = ast.getRootNode();

  auto getNthAncestorLogicalExpr = [&](int N) -> Node* {
    // find first LogicalExpr in AST
    std::pair<Node*, int> candidate(nullptr, 0);
    std::queue<Node*> q{{rootNode}};
    while (!q.empty()) {
      auto curNode = q.front();
      q.pop();
      if (dynamic_cast<LogicalExpr*>(curNode)) {
        candidate = std::make_pair(curNode, candidate.second + 1);
        if (candidate.first != nullptr && candidate.second == N)
          return candidate.first;
      }
      // add all parent nodes
      std::for_each(curNode->getParents().begin(), curNode->getParents().end(), [&](Node* n) { q.push(n); });
    }
    return nullptr;
  };

  // make sure that v is a LogicalExpr, otherwise find first (N=1) LogicalExpr by traversing parents
  if (dynamic_cast<LogicalExpr*>(rootNode) == nullptr) {
    startNode = getNthAncestorLogicalExpr(1);
    if (!startNode) throw std::logic_error("AST does not contain any LogicalExpr!");
  }

  // compute minDepth required by getReducibleCones algorithm
  int minDepth = computeMinDepth(startNode);

  // v has no non-critical input node p -> return empty set
  if (minDepth == -1) return std::vector<Node*>();

  std::cout << "getReducibleCones(" << startNode->getUniqueNodeId() << ", " << minDepth << ")" << std::endl;
  return getReducibleCones(startNode, minDepth);
}



// ------------------------------------–
// Internal (non-universal) helper methods
// ------------------------------------–

int ConeRewriter::computeMinDepth(Node* v) {
  // find a non-critical input node p of v
  auto isNoOperatorNode = [](Node* n) { return (dynamic_cast<Operator*>(n) == nullptr); };
  int lMax = v->getMultDepthL(multiplicativeDepths);
  for (auto &p : v->getParentsNonNull()) {
    // exclude Operator nodes as they do not have any parent and are not modeled as node in the paper
    if (isNoOperatorNode(p) && !isCriticalNode(lMax, p)) {
      // set minMultDepth as l(p)+2 and call getReducibleCones
      return p->getMultDepthL(multiplicativeDepths) + 2;
      // According to the paper (see p. 9, §2.4) minMultDepth = l(p)+1 is used but it does not return any result:
      // return p->getMultDepthL(multiplicativeDepthOfNode) + 1;
    }
  }
  // return -1 (error) if node v has no non-critical input node
  return -1;
}

std::vector<Node*> ConeRewriter::getReducibleCones(Node* v, int minDepth) {
  // return empty set if minDepth is reached
  if (v->getMultDepthL(multiplicativeDepths) == minDepth) return std::vector<Node*>();

  // get critical predecessor nodes: { p ∈ pred(v) | l(p) = l(v) - d(v) }
  std::vector<Node*>* P = getCriticalPredecessors(v);

  // return v if at least one predecessor of v is non-critical and v is an AND-gate
  auto logicalExp = dynamic_cast<LogicalExpr*>(v);
  if (P->size() < 2 && logicalExp != nullptr && logicalExp->getOp()->equals(OpSymb::logicalAnd)) {
    // return set consisting of start node v only
    return std::vector<Node*>{v};
  }

  // determine reducible input cones
  std::vector<std::vector<Node*>> deltaR;
  std::vector<Node*> delta;
  for (auto &p : *P) {
    std::vector<Node*> intermedResult = getReducibleCones(p, computeMinDepth(p));
    if (!intermedResult.empty()) deltaR.push_back(intermedResult);
  }

  // return empty set if either one of the following is true:
  // a. v is not a LogicalExpr
  // b. v is a LogicalExpr but not an AND- or XOR-gate
  // b. v is an AND-gate and deltaR is empty
  // c. v is a XOR-gate and size of deltaR does not equal size of P
  if (logicalExp == nullptr ||
      !(logicalExp->getOp()->equals(OpSymb::logicalAnd) || logicalExp->getOp()->equals(OpSymb::logicalXor)) ||
      (logicalExp->getOp()->equals(OpSymb::logicalAnd) && deltaR.empty()) ||
      (logicalExp->getOp()->equals(OpSymb::logicalXor) && deltaR.size() != P->size())) {
    return std::vector<Node*>();
  }

  if (logicalExp->getOp()->equals(OpSymb::logicalAnd)) {
    // both cones must be reducible because deltaR is non-empty -> pick a random one, and assign to delta
    std::cout << "> both cones must be reducible, picking randomly from sets: { ";
    for (auto &vec : deltaR) std::cout << "  " << vec;
    std::cout << " }" << std::endl;
    delta = *select_randomly(deltaR.begin(), deltaR.end());
  } else if (logicalExp->getOp()->equals(OpSymb::logicalXor)) {
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

int ConeRewriter::getMaxMultDepth(Ast &inputAst) {
  std::queue<Node*> nodesToCheck;
  nodesToCheck.push(inputAst.getRootNode());
  int highestDepth = 0;
  while (!nodesToCheck.empty()) {
    auto curNode = nodesToCheck.front();
    nodesToCheck.pop();
    highestDepth = std::max(highestDepth, curNode->getMultDepthL(multiplicativeDepths));
    auto vec = inputAst.isReversed() ? curNode->getParents() : curNode->getChildren();
    std::for_each(vec.begin(), vec.end(), [&](Node* n) {
      nodesToCheck.push(n);
    });
  }
  return highestDepth;
}

void ConeRewriter::addElements(std::vector<Node*> &result, std::vector<Node*> newElements) {
  result.reserve(result.size() + newElements.size());
  result.insert(result.end(), newElements.begin(), newElements.end());
}

void ConeRewriter::flattenVectors(std::vector<Node*> &resultVector, std::vector<std::vector<Node*>> vectorOfVectors) {
  std::set<Node*> res;
  std::for_each(vectorOfVectors.begin(), vectorOfVectors.end(), [&](std::vector<Node*> rVec) {
    res.insert(rVec.begin(), rVec.end());
    //resultVector.insert(resultVector.end(), rVec.begin(), rVec.end());
  });
  resultVector.assign(res.begin(), res.end());
}

std::vector<Node*>* ConeRewriter::getCriticalPredecessors(Node* v) {
  // P <- { p ∈ pred(v) | l(p) = l(v) - d(v) }
  auto result = new std::vector<Node*>();
  int criterion = v->getMultDepthL(multiplicativeDepths) - v->depthValue();
  for (auto &p : v->getParentsNonNull()) {
    if (p->getMultDepthL(multiplicativeDepths) == criterion) result->push_back(p);
  }
  return result;
}

void ConeRewriter::reverseEdges(const std::vector<Node*> &nodes) {
  for (Node* n : nodes) {
    n->swapChildrenParents();
  }
}

void ConeRewriter::getReducibleConesForEveryPossibleStartingNode(Ast &ast) {
  for (Node* n : ast.getRootNode()->getAnc()) {
    int minDepth = computeMinDepth(n);
    if (minDepth == -1) continue;
    std::cout << "ConeRewriter::getReducibleCones(" << n->getUniqueNodeId() << ", " << minDepth << ")" << std::endl;
    std::vector<Node*> delta = getReducibleCones(n, minDepth);
    std::cout << "  delta: " << std::endl << "  " << delta << std::endl;

    // cAndCkt: C^{AND} circuit consisting of critical AND-nodes that are connected if there is a multiplicative depth-2
    // path in between two of those nodes in the initial circuit
    // each cone δ ∈ Δ has an unique terminal AND node in C^{AND}
    std::vector<Node*> cAndCkt = ConeRewriter::getAndCriticalCircuit(delta);
    std::cout << "  cAndCkt:" << std::endl << "  " << cAndCkt << std::endl;

    // deltaMin: the minimum set of reducible cones
    std::vector<Node*> deltaMin = ConeRewriter::selectCones(cAndCkt);
    std::cout << "  deltaMin: " << std::endl << "  " << deltaMin << std::endl << std::endl;
  }
  exit(0);
}

bool ConeRewriter::isCriticalNode(int lMax,
                                  Node* n,
                                  std::map<std::string, int>* multDepthsMap,
                                  std::map<std::string, int>* reverseMultDepthsMap) {
  int l = n->getMultDepthL(multDepthsMap);
  int r = n->getReverseMultDepthR(reverseMultDepthsMap);
  return (lMax == l + r);
}

bool ConeRewriter::isCriticalNode(int lMax, Node* n) {
  return isCriticalNode(lMax, n, multiplicativeDepths, multiplicativeDepthsReversed);
}

// This implementation of getAndCriticalCircuit does not rely on the computation of delta, but instead explores the
// AST by itself while checking which nodes are critical.
//
//std::vector<Node*> ConeRewriter::getAndCriticalCircuit(const Ast &ast) {
//  // find all critical AND nodes
//  // - determine max multiplicative depth l_{max}
//  int lMax = getMaxMultDepth(ast);
//
//  // - find nodes for which l_{max} = l(v) + r(v) holds -> critical nodes
////  std::cout << "AST_{criticalNodes}:" << std::endl << "  [";
//  std::unordered_map<std::string, Node*> criticalNodes;
//  for (auto &node : getAnc(ast.getRootNode())) {
//    auto lexp = dynamic_cast<LogicalExpr*>(node);
//    if (lexp != nullptr && lexp->getOp().equals(OpSymb::logicalAnd)) {
//      if (ConeRewriter::isCriticalNode(lMax, node)) {
//        criticalNodes.emplace(node->getUniqueNodeId(), node);
////        std::cout << node->getUniqueNodeId() << ", ";
//      }
//    }
//  }
////  std::cout << "\b\b]" << std::endl;
//
//  // duplicate critical nodes to create new circuit C_{AND} as we do not want to modify the original circuit
//  // note that clone() does not copy the links to parents and children
//  std::map<std::string, Node*> cAndMap;
//  std::vector<Node*> cAndResultCkt;
//  for (auto &[k, v] : criticalNodes) {
//    auto clonedNode = v->clone();
//    clonedNode->setUnderlyingNode(v);
//    cAndMap.emplace(v->getUniqueNodeId(), clonedNode);
//    cAndResultCkt.push_back(clonedNode);
//  }
//
//  // check if there are depth-2 critical paths in between critical nodes in the original ckt
//  for (auto &[k, v] : criticalNodes) {
//    std::queue<Node*> q{{v}};
//    while (!q.empty()) {
//      auto curNode = q.front();
//      q.pop();
//      // normally the node should only have exactly one child; however, we consider the general case of N>1 children
//      for (auto &child : curNode->getChildren()) {
//        auto childLexp = dynamic_cast<LogicalExpr*>(child);
//        // if the child is a LogicalExpr of type AND-gate
//        if (childLexp != nullptr && childLexp->getOp().equals(OpSymb::logicalAnd)) {
//          // check if this child is a critical node, if yes: connect both nodes
//          if (criticalNodes.find(childLexp->getUniqueNodeId()) != criticalNodes.end()) {
//            Node* copiedV = cAndMap[v->getUniqueNodeId()];
//            Node* copiedChild = cAndMap[child->getUniqueNodeId()];
//            copiedV->addChild(copiedChild);
//            copiedChild->addParent(copiedV);
//            // std::cout << "added edge: { " << copiedV->getUniqueNodeId() << " -> " << copiedChild->getUniqueNodeId()
//            // << " }" << std::endl;
//          }
//        } else {  // continue if the child is not a LogicalExpr --> node does not influence the multiplicative depth
//          q.push(child);
//        }
//      }
//    }
//  }
//  return cAndResultCkt;
//}

std::vector<Node*> ConeRewriter::getAndCriticalCircuit(std::vector<Node*> delta) {
  // remove non-AND nodes from delta (note: delta is passed as copy-by-value) as delta may also include XOR nodes
  delta.erase(remove_if(delta.begin(), delta.end(), [](Node* d) {
    auto lexp = dynamic_cast<LogicalExpr*>(d);
    return (lexp == nullptr || !lexp->getOp()->equals(OpSymb::logicalAnd));
  }), delta.end());

  // duplicate critical nodes to create new circuit C_{AND} as we do not want to modify the original circuit
  std::map<std::string, Node*> cAndMap;
  std::vector<Node*> cAndResultCkt;
  for (auto &v : delta) {
    // note that clone() does not copy the links to parents and children
    auto clonedNode = v->clone();
    // a back-link to the node in the original circuit
    clonedNode->setUnderlyingNode(v);
    cAndMap.emplace(v->getUniqueNodeId(), clonedNode);
    cAndResultCkt.push_back(clonedNode);
  }

  // in case that there are less than two nodes, we can not connect any two nodes
  if (delta.size() < 2) return cAndResultCkt;

  // check if there are depth-2 critical paths in between critical nodes in the original ckt
  for (auto &v : delta) {
    std::queue<Node*> q{{v}};
    while (!q.empty()) {
      auto curNode = q.front();
      q.pop();
      for (auto &child : curNode->getChildren()) {
        auto childLexp = dynamic_cast<LogicalExpr*>(child);
        // if the child is a LogicalExpr of type AND-gate
        if (childLexp != nullptr && childLexp->getOp()->equals(OpSymb::logicalAnd)) {
          // check if this child is a critical node, if yes: connect both nodes
          if (std::find(delta.begin(), delta.end(), childLexp) != delta.end()) {
            Node* copiedV = cAndMap[v->getUniqueNodeId()];
            Node* copiedChild = cAndMap[child->getUniqueNodeId()];
            copiedV->addChild(copiedChild);
            copiedChild->addParent(copiedV);
            // std::cout << "added edge: { " << copiedV->getUniqueNodeId() << " -> "
            // << copiedChild->getUniqueNodeId() << " }" << std::endl;
          }
        } else {  // continue if child is not a LogicalExpr --> node does not influence the mult. depth
          q.push(child);
        }
      }
    }
  }
  return cAndResultCkt;
}

/// This method implements Kahn's algorithm as described at wikipedia,
/// see https://en.wikipedia.org/wiki/Topological_sorting#CITEREFKahn1962
std::vector<Node*> ConeRewriter::sortTopologically(const std::vector<Node*> &nodes) {
  std::vector<Node*> L;
  std::map<Node*, int> numEdgesDeleted;

  // S <- nodes without an incoming edge
  std::vector<Node*> S;
  std::for_each(nodes.begin(), nodes.end(), [&](Node* n) {
    if (n->getParents().empty()) S.push_back(n);
  });

  while (!S.empty()) {
    auto n = S.back();
    S.pop_back();
    L.push_back(n);
    for (auto &m : n->getChildrenNonNull()) {
      numEdgesDeleted[m] += 1; // emulates removing edge from the graph
      if (m->getParents().size() == numEdgesDeleted[m]) S.push_back(m);
    }
  }
  return L;
}

std::vector<Node*> ConeRewriter::selectCones(std::vector<Node*> cAndCkt) {
  /// Helper function that computes the flow of all nodes in the given circuit ckt
  auto computeFlow = [](std::vector<Node*> &ckt) {
    std::map<Node*, float> computedFlow;
    std::map<std::pair<Node*, Node*>, float> edgeFlows;
    auto topologicalOrder = ConeRewriter::sortTopologically(ckt);
    for (Node* v : topologicalOrder) {
      if (v->getParentsNonNull().empty()) {  // if v is input
        // trivial flow of 1
        computedFlow[v] = 1;
      } else {  // if v is intermediate node
        // compute flow by accumulating flows of incoming edges (u,v) where u ∈ pred(v)
        auto predecessorsOfV = v->getParentsNonNull();
        float flow = 0.0f;
        std::for_each(predecessorsOfV.begin(), predecessorsOfV.end(), [&](Node* u) {
          flow += edgeFlows[std::pair(u, v)];
        });
        computedFlow[v] = flow;
      }
      // for all successors u: define edge flow
      for (auto &u : v->getChildrenNonNull()) {
        edgeFlows[std::make_pair(v, u)] = computedFlow[v] / v->getChildrenNonNull().size();
      }
    }
    return computedFlow;
  };

  // ----------------------------

  std::vector<Node*> deltaMin;
  while (!cAndCkt.empty()) {
    // compute all node flows f^{+}(v) in cAndCkt
    std::map<Node*, float> nodeFlows = computeFlow(cAndCkt);

    // reverse circuit edges and compute all ascending node flows f^{-}(v)
    ConeRewriter::reverseEdges(cAndCkt);
    std::map<Node*, float> nodeFlowsAscending = computeFlow(cAndCkt);

    // compute f(v) for all nodes
    std::map<Node*, float> flows;
    for (auto &n : cAndCkt) flows[n] = nodeFlowsAscending[n] * nodeFlows[n];

    // find the node with the largest flow u
    Node* u = std::max_element(
        flows.begin(),
        flows.end(),
        [](const std::pair<Node*, float> &p1, const std::pair<Node*, float> &p2) {
          return p1.second < p2.second;
        })->first;

    // remove node u
    // - remove any edges pointing to node u as child
    for (auto &p : u->getParentsNonNull()) p->removeChild(u);

    // - remove any edges pointing to node u as parent
    for (auto &p : u->getChildrenNonNull()) p->removeParent(u);

    // - remove node u from cAndCkt
    cAndCkt.erase(std::remove(cAndCkt.begin(), cAndCkt.end(), u), cAndCkt.end());

    // add critical cone ending at node u to deltaMin
    deltaMin.push_back(u);
  }

  return deltaMin;
}

Node* ConeRewriter::getNonCriticalLeafNode() {
  Node* candidateNode = nullptr;
  const int maxDepth = getMaxMultDepth(ast);
  std::queue<Node*> q{{ast.getRootNode()}};
  while (!q.empty()) {
    auto curNode = q.front();
    q.pop();
    for (auto &p : curNode->getParents()) {
      // remember found non-critical leaf node
      if (p->getParents().empty()
          && maxDepth != (p->getMultDepthL(multiplicativeDepths)
              + p->getReverseMultDepthR(multiplicativeDepthsReversed))) {
        candidateNode = p;
      }
      // enqueue the parent nodes and continue
      q.push(p);
    }
  }
  // returns the non-critical leaf node with the highest tree-depth
  return candidateNode;
}

void ConeRewriter::printConesAsGraphviz(std::vector<Node*> &nodes) {
  // print each cone as separate digraph in dot format
  for (auto &coneStartingNode : nodes) {
    std::cout << "digraph D {" << std::endl;
    // we need to take the node in the underlying AST
    auto uNode = coneStartingNode->getUnderlyingNode();
    std::set<Node*> printedNodes;
    std::queue<Node*> q{{uNode}};
    while (!q.empty()) {
      auto curNode = q.front();
      q.pop();
      // make sure that every node is only printed once
      if (printedNodes.count(curNode) > 0) continue;
      std::cout << DotPrinter::getDotFormattedString(curNode, "\t", true) << std::endl;
      printedNodes.insert(curNode);
      // only consider curNode's children if the node is the starting node or not an AND node (= end of cone)
      auto curNodeLexp = dynamic_cast<LogicalExpr*>(curNode);
      if (curNode == uNode || (curNodeLexp != nullptr && !curNodeLexp->getOp()->equals(OpSymb::logicalAnd))) {
        for (auto &p : curNode->getParents()) q.push(p);
      }
    }
    std::cout << "}" << std::endl;
  }
}

std::vector<Node*> ConeRewriter::rewriteMultiInputGate(std::vector<Node*> inputNodes, OpSymb::LogCompOp gateType) {
  // Transforms a multi-input gate taking n inputs into a sequence of binary gates
  // For example, consider a n-input logical AND (&):
  //   &_{i=1}^{n} y_1, y_2, y_3, ..., y_m
  // that is transformed into the expression
  //   ((((y_1 & y_2) & y_3) ...) & y_m)

  std::vector<Node*> outputNodes;
  auto it = std::begin(inputNodes);

  // handle first "special" gate
  auto left = dynamic_cast<AbstractExpr*>(*it);
  assert(left != nullptr);
  ++it;
  auto right = dynamic_cast<AbstractExpr*>(*it);
  assert(right != nullptr);
  ++it;
  auto recentLexp = new LogicalExpr(left, gateType, right);
  outputNodes.push_back(recentLexp);

  // handle all other gates
  for (auto end = std::end(inputNodes); it != end; ++it) {
    auto newLexp = new LogicalExpr(recentLexp, gateType, dynamic_cast<AbstractExpr*>(*it));
    outputNodes.push_back(newLexp);
    recentLexp = newLexp;
  }

  return outputNodes;
}

std::pair<Node*, Node*> ConeRewriter::getCriticalAndNonCriticalInput(int lMax, Node* n) {
  return getCriticalAndNonCriticalInput(lMax, n, multiplicativeDepths, multiplicativeDepthsReversed);
}

void ConeRewriter::rewriteCones(Ast &astToRewrite, const std::vector<Node*> &coneEndNodes) {
  std::cout << ">> Running rewriteCones..." << std::endl;
  int maxMultDepth = getMaxMultDepth(astToRewrite);

  // reverse back the AST that has been reversed previously to facilitate the cone selection algorithms.
  // as we need to, however, modify the graph now, we need the "original" order again to not break anything
  auto multDepths = new std::map<std::string, int>();
  auto reverseMultDepths = new std::map<std::string, int>();
  assert(astToRewrite.isReversed());
  DotPrinter::printAsDotFormattedGraph(astToRewrite);
  if (astToRewrite.isReversed()) {
    astToRewrite.reverseEdges();
    precomputeMultDepths(astToRewrite, multDepths, reverseMultDepths);
  }

  // Assumption: δ ∈ coneEndNodes represents a cone that ends at node δ
  for (auto coneEnd : coneEndNodes) {
    // we need to get the node in the underlying circuit as C^{AND} only contains a limited subset of nodes
    coneEnd = coneEnd->getUnderlyingNode();

    // determine bounds of the cone
    // -- upper bound: parent node of cone end
    auto rNode = coneEnd->getParentsNonNull().front();
    assert(rNode->getChildrenNonNull().size() == 1);

    // U_y, the multi-input XOR, is represented as multiple XOR nodes in our tree:
    // coneEndNodes [v_1, ..., v_n] --> xorN --> xorN-1 ---> xorN-2 ---> ... ---> xor1 --> sNode v_t.
    // We denote xorN as xorEndNode and xor1 as xorStartNode. We know that xorStartNode must be the first node in the
    // cone, i.e., the first child of the cone's end node.
    auto[xorStartNode, a_t] = getCriticalAndNonCriticalInput(maxMultDepth, coneEnd, multDepths, reverseMultDepths);
    // we "cut-off" call edges from node a_t in order to reconnect node later
    a_t->isolateNode();

    // -- lower bound: first AND node while following critical path
    // find the ancestors of δ that are LogicalExpr --> start nodes (v_1, ..., v_n) of cone
    std::vector<Node*> coneStartNodes;
    std::queue<Node*> q{{coneEnd}};
    while (!q.empty()) {
      auto curNode = q.front();
      q.pop();
      auto curNodeLexp = dynamic_cast<LogicalExpr*>(curNode);
      // if curNode is not the cone end node and a logical-AND expression, we found one of the end nodes
      // in that case we can stop exploring this path any further
      if (curNode != coneEnd && curNodeLexp != nullptr && curNodeLexp->getOp()->equals(OpSymb::logicalAnd)) {
        coneStartNodes.push_back(curNode);
      } else {  // otherwise we need to continue search by following the critical path
        // add parent nodes of current nodes -> continue BFS traversal
        for (auto &child : curNode->getChildrenNonNull()) {
          if (isCriticalNode(maxMultDepth, child, multDepths, reverseMultDepths)) q.push(child);
        }
      }
    }

    std::vector<Node*> finalXorInputs;
    // It should not make a difference which of the cone start nodes we take - all of them should have the same parent.
    Node* xorEndNode = coneStartNodes.front()->getParentsNonNull().front();
    for (auto &startNode : coneStartNodes) assert(startNode->getParentsNonNull().front() == xorEndNode);

    // collect all non-critical inputs y_1, ..., y_m in between xorStartNode up to xorEndNode
    std::vector<Node*> inputsY1ToYm;
    auto currentNode = xorStartNode;
    while (currentNode != xorEndNode) {
      auto[critIn, nonCritIn] = getCriticalAndNonCriticalInput(maxMultDepth, currentNode);
      currentNode->removeChild(nonCritIn);
      nonCritIn->removeParent(currentNode);
      inputsY1ToYm.push_back(nonCritIn);
      currentNode = critIn;
    }

    // XorEndNode: remove incoming edges from nodes v_1, ..., v_n
    for (auto &p : xorEndNode->getChildrenNonNull()) {
      // keep the operator node -> only remove those nodes that are no operators
      if (dynamic_cast<Operator*>(p) == nullptr) xorEndNode->removeChildBilateral(p);
    }

    // XorStartNode: remove incoming edge from xorStartNode to v_t
    for (auto &c : xorStartNode->getParentsNonNull()) {
      c->removeChildBilateral(xorStartNode);
    }

    LogicalExpr* u_y = nullptr;
    if (inputsY1ToYm.size() == 1) {  // if y_1 only exists: connect input y_1 directly to u_y -> trivial case
      u_y = new LogicalExpr(static_cast<AbstractExpr*>(a_t),
                            OpSymb::logicalAnd,
                            static_cast<AbstractExpr*>(inputsY1ToYm.front()));
    } else if (inputsY1ToYm.size() > 1) {  // otherwise there are inputs y_1, y_2, ..., y_m
      // TODO check that this branch works
      // otherwise build XOR chain of inputs and connect last one as input of u_y
      std::vector<Node*> yXorChain = ConeRewriter::rewriteMultiInputGate(inputsY1ToYm, OpSymb::logicalXor);
      std::for_each(yXorChain.begin(), yXorChain.end(), [&](Node* n) { n->swapChildrenParents(); });
      u_y = new LogicalExpr(a_t, OpSymb::logicalAnd, yXorChain.back());
    } else {
      throw std::logic_error("Unexpected number (0) of non-critical inputs y_1, ..., y_m.");
    }
    finalXorInputs.push_back(u_y);

    // for each of these start nodes v_i
    for (auto sNode : coneStartNodes) {
      auto sNodeLexp = dynamic_cast<LogicalExpr*>(sNode);
      assert(sNodeLexp != nullptr);

      // determine critical input a_1^i and non-critical input a_2^i of v_i
      auto[criticalInput, nonCriticalInput] = getCriticalAndNonCriticalInput(maxMultDepth, sNode);

      // remove critical input of v_i
      criticalInput->removeParent(sNode);
      sNode->removeChild(criticalInput);

      // remove all outgoing edges of v_i
      sNode->removeChildBilateral(criticalInput);

      // add non-critical input of v_t (coneEnd) named a_t as input to v_i
      auto originalOperator = sNodeLexp->getOp();
      sNodeLexp->setAttributes(dynamic_cast<AbstractExpr*>(nonCriticalInput),
                               originalOperator,
                               dynamic_cast<AbstractExpr*>(a_t));

      // create new logical-AND node u_i and set v_i as input of u_i
      auto leftOp = dynamic_cast<AbstractExpr*>(criticalInput);
      auto uNode = new LogicalExpr(leftOp, OpSymb::logicalAnd, sNodeLexp);
      finalXorInputs.push_back(uNode);
    }

    // convert multi-input XOR into binary XOR nodes
    std::vector<Node*> xorFinalGate = ConeRewriter::rewriteMultiInputGate(finalXorInputs, OpSymb::logicalXor);
    for (auto &gate : xorFinalGate) { gate->getUniqueNodeId(); }

    // remove coneEnd
    coneEnd = astToRewrite.deleteNode(coneEnd, true);
    assert(coneEnd == nullptr);

    // connect final XOR (i.e., last LogicalExpr in the chain of XOR nodes) to cone output
    rNode->addChild(xorFinalGate.back());
    xorFinalGate.back()->addParent(rNode);

    std::cout << "----" << std::endl;
    DotPrinter::printAllReachableNodes(rNode);
    std::cout << "----" << std::endl;

    delete multDepths;
    delete reverseMultDepths;

    std::cerr << "rewriteCones not implemented yet! Aborting..." << std::endl;
    exit(1);
  } // for (auto coneEnd : coneEndNodes)
  // return optimizedAst;
}

void ConeRewriter::precomputeMultDepths() {
  precomputeMultDepths(ast, multiplicativeDepths, multiplicativeDepthsReversed);
}

void ConeRewriter::precomputeMultDepths(Ast &inputAst,
                                        std::map<std::string, int>* multDepthsMap,
                                        std::map<std::string, int>* reverseMultDepthsMap) {
  // precompute the AST's multiplicative depth and reverse multiplicative depth
  multDepthsMap->clear();
  reverseMultDepthsMap->clear();
  for (auto &node : inputAst.getAllNodes()) {
    node->getMultDepthL(multDepthsMap);
    node->getReverseMultDepthR(reverseMultDepthsMap);
  }

  // determine and store the AST's maximum multiplicative depth
  maximumMultiplicativeDepth = std::max_element(
      multDepthsMap->begin(), multDepthsMap->end(),
      [](const std::pair<const std::basic_string<char>, int> &a,
         const std::pair<const std::basic_string<char>, int> &b) {
        return a.second < b.second;
      })->second;

  // compute "identity" of AST by generating GraphViz output and hashing it
  std::cout << "" << std::endl;
}

ConeRewriter::ConeRewriter(Ast &ast) : ast(ast), maximumMultiplicativeDepth(0) {
  multiplicativeDepths = new std::map<std::string, int>();
  multiplicativeDepthsReversed = new std::map<std::string, int>();
}

ConeRewriter::~ConeRewriter() {
  delete multiplicativeDepths;
  delete multiplicativeDepthsReversed;
}

std::pair<Node*, Node*> ConeRewriter::getCriticalAndNonCriticalInput(int lMax,
                                                                     Node* n,
                                                                     std::map<std::string, int>* multDepthsMap,
                                                                     std::map<std::string, int>* reverseMultDepthsMap) {
  auto node = dynamic_cast<LogicalExpr*>(n);
  bool leftIsCritical = isCriticalNode(lMax, node->getLeft(), multDepthsMap, reverseMultDepthsMap);
  bool rightIsCritical = isCriticalNode(lMax, node->getRight(), reverseMultDepthsMap, reverseMultDepthsMap);
  if (leftIsCritical && rightIsCritical) {
    throw std::invalid_argument("Cannot rewrite given AST because input of cone's end node are both critical!");
  } else if (!leftIsCritical && !rightIsCritical) {
    throw std::invalid_argument("Neither input left nor input right are critical nodes!");
  }

  return (leftIsCritical ? std::make_pair(node->getLeft(), node->getRight())
                         : std::make_pair(node->getRight(), node->getLeft()));
}
