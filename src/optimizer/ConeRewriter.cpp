#include <LogicalExpr.h>
#include <set>
#include <queue>
#include "ConeRewriter.h"
#include "Return.h"
#include "Function.h"


// ------------------------------------–
// Algorithms presented in the paper
// ------------------------------------–

Ast &ConeRewriter::applyConeRewriting(Ast &ast) {
  // Ensure that we have a circuit of supported nodes only
  if (!ast.isValidCircuit()) {
    throw std::invalid_argument(
        "Cannot perform Cone Rewriting on AST because AST consists of Node types that are not circuit-compatible. "
        "These nodes do not define the required child/parent relationship of class Node.");
  }

  // We need to reverse the tree's edges to be conform to the paper
  if (!ast.isReversed()) {
    ast.reverseEdges();
  }
  ast.printGraphviz();
  return ConeRewriter::applyMinMultDepthHeuristic(ast);
}

// TODO this method has not been tested yet as rewriteCones(...) is not fully working yet
Ast &ConeRewriter::applyMinMultDepthHeuristic(Ast &ast) {
  // create a deep copy of the input circuit
  Ast* optimizedCkt = new Ast(ast);

  // compute reducible cones set
  std::vector<Node*> deltaMin = ConeRewriter::computeReducibleCones(ast);

  // repeat rewriting as long as there are any reducible cones
  // optimizedCkt will contain all rewritten cones
  while (!deltaMin.empty()) {
    // rewrite all cones from deltaMin in the input circuit
    // we need to take the input circuit because the node refs in the cones refer to nodes in the input circuit
    ConeRewriter::rewriteCones(ast, deltaMin);

    // if depth of cone-rewritten circuit is smaller -> this is our new optimized circuit
    if (getMaxMultDepth(ast) < getMaxMultDepth(*optimizedCkt)) {
      optimizedCkt = &ast;
    }

    // recompute reducible cones
    deltaMin = ConeRewriter::computeReducibleCones(ast);
  }

  // return multiplicative depth-optimized circuit
  return *optimizedCkt;
}

std::vector<Node*> ConeRewriter::computeReducibleCones(Ast &ast) {
  // delta: set of all reducible cones
  std::vector<Node*> delta = ConeRewriter::getReducibleCones(ast);
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

std::vector<Node*> ConeRewriter::getReducibleCones(Ast &ast) {
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
  int lMax = v->getMultDepthL();
  for (auto &p : v->getPred()) {
    // exclude Operator nodes as they do not have any parent and are not modeled as node in the paper
    if (isNoOperatorNode(p) && !isCriticalNode(lMax, p)) {
      // set minMultDepth as l(p)+2 and call getReducibleCones
      return p->getMultDepthL() + 2;
      // According to the paper (see p. 9, §2.4) minMultDepth = l(p)+1 is used but it does not return any result:
      // return getMultDepthL(p) + 1;
    }
  }
  // return -1 (error) if node v has no non-critical input node
  return -1;
}

std::vector<Node*> ConeRewriter::getReducibleCones(Node* v, int minDepth) {
  // return empty set if minDepth is reached
  if (v->getMultDepthL() == minDepth) return std::vector<Node*>();

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

int ConeRewriter::getMaxMultDepth(const Ast &ast) {
  std::queue<Node*> nodesToCheck;
  nodesToCheck.push(ast.getRootNode());
  int highestDepth = 0;
  while (!nodesToCheck.empty()) {
    auto curNode = nodesToCheck.front();
    nodesToCheck.pop();
    highestDepth = std::max(highestDepth, curNode->getMultDepthL());
    auto vec = ast.isReversed() ? curNode->getParents() : curNode->getChildren();
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
  int criterion = v->getMultDepthL() - v->depthValue();
  for (auto &p : v->getPred()) {
    if (p->getMultDepthL() == criterion) result->push_back(p);
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
    std::vector<Node*> delta = ConeRewriter::getReducibleCones(n, minDepth);
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

bool ConeRewriter::isCriticalNode(int lMax, Node* n) {
  int l = n->getMultDepthL();
  int r = n->getReverseMultDepthR();
  return (lMax == l + r);
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
    for (auto &m : n->getChildren()) {
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
      if (v->getParents().empty()) {  // if v is input
        // trivial flow of 1
        computedFlow[v] = 1;
      } else {  // if v is intermediate node
        // compute flow by accumulating flows of incoming edges (u,v) where u ∈ pred(v)
        auto predecessorsOfV = v->getPred();
        float flow = 0.0f;
        std::for_each(predecessorsOfV.begin(), predecessorsOfV.end(), [&](Node* u) {
          flow += edgeFlows[std::pair(u, v)];
        });
        computedFlow[v] = flow;
      }
      // for all successors u: define edge flow
      for (auto &u : v->getSucc()) {
        edgeFlows[std::make_pair(v, u)] = computedFlow[v] / v->getSucc().size();
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
    std::for_each(cAndCkt.begin(), cAndCkt.end(), [&](Node* n) {
      flows[n] = nodeFlowsAscending[n] * nodeFlows[n];
    });

    // find the node with the largest flow u
    Node* u = std::max_element(
        flows.begin(),
        flows.end(),
        [](const std::pair<Node*, float> &p1, const std::pair<Node*, float> &p2) {
          return p1.second < p2.second;
        })->first;

    // remove node u
    // - remove any edges pointing to node u as child
    for (auto &p : u->getParents()) {
      p->removeChild(u);
    }
    // - remove any edges pointing to node u as parent
    for (auto &p : u->getChildren()) {
      p->removeParent(u);
    }
    // - remove node u from cAndCkt
    cAndCkt.erase(std::remove(cAndCkt.begin(), cAndCkt.end(), u), cAndCkt.end());

    // add critical cone ending at node u to deltaMin
    deltaMin.push_back(u);
  }

  return deltaMin;
}

Node* ConeRewriter::getNonCriticalLeafNode(Ast &ast) {
  Node* candidateNode = nullptr;
  const int maxDepth = getMaxMultDepth(ast);
  std::queue<Node*> q{{ast.getRootNode()}};
  while (!q.empty()) {
    auto curNode = q.front();
    q.pop();
    for (auto &p : curNode->getParents()) {
      // remember found non-critical leaf node
      if (p->getParents().empty() && maxDepth != (p->getMultDepthL() + p->getReverseMultDepthR())) {
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
      std::cout << curNode->getDotFormattedString(true, "\t", true) << std::endl;
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

  for (auto &node : outputNodes) node->swapChildrenParents();

  return outputNodes;
}

std::pair<Node*, Node*> getCriticalAndNonCriticalInput(Node* n) {
  auto node = dynamic_cast<LogicalExpr*>(n);
  if (node->getLeft()->getMultDepthL() > node->getRight()->getMultDepthL()) {
    return std::make_pair(node->getLeft(), node->getRight());
  } else {
    return std::make_pair(node->getRight(), node->getLeft());
  }
}

void ConeRewriter::rewriteCones(Ast &ast, const std::vector<Node*> &coneEndNodes) {
  std::cout << ">> Running rewriteCones..." << std::endl;

  // reverse back the AST that has been reversed in order to facilitate the cone selection algorithms
  if (ast.isReversed()) ast.reverseEdges();

  // TODO adapt algorithm below after reversing edges back
  ast.printGraphviz();

  // Assumption: δ ∈ coneEndNodes represents a cone that ends at node δ
  for (auto coneEnd : coneEndNodes) {
    // we need to get the node in the underlying circuit as C^{AND} only contains a limited subset of nodes
    coneEnd = coneEnd->getUnderlyingNode();

    // get the node that is the child node of the cone's end
    auto rNode = coneEnd->getChildren().front();
    assert(rNode->getChildren().size() == 1);

    auto[xorStartNode, a_t] = getCriticalAndNonCriticalInput(coneEnd);
    for (auto &c : a_t->getChildren()) c->removeParent(a_t);
    a_t->removeChildren();

    // find the ancestors of δ that are LogicalExpr --> start nodes (v_1, ..., v_n) of cone
    std::vector<Node*> coneStartNodes;
    std::queue<Node*> q{{coneEnd}};
    while (!q.empty()) {
      auto curNode = q.front();
      q.pop();
      auto curNodeLexp = dynamic_cast<LogicalExpr*>(curNode);
      if (curNode != coneEnd && curNodeLexp != nullptr && curNodeLexp->getOp()->equals(OpSymb::logicalAnd)) {
        // stop traversing this branch further -> we found one of possibly multiple start nodes of the cone
        coneStartNodes.push_back(curNode);
      } else {
        // add parent nodes of current nodes -> continue BFS traversal
        for (auto &child : curNode->getParents()) q.push(child);
      }
    }

    // works: printAllNodesAsGraphviz(coneEnd->getChildren().front());

    std::vector<Node*> finalXorInputs;
    // U_y, the multi-input XOR, is represented as multiple XOR nodes in our tree
    // coneEndNodes [v_1, ..., v_n] --> xorEndNode --> xorN ---> xorN-1 ---> ... ---> xor1 --> xorStartNode --> sNode v_t
    Node* xorEndNode = coneStartNodes.front()->getChildren().front(); // TODO check this..

    // collect all non-critical inputs xInp in between xorStartNode up to xorEndNode
    std::vector<Node*> inputsY1ToYm;
    auto currentNode = xorStartNode;
    while (currentNode != xorEndNode) {
      auto[critIn, nonCritIn] = getCriticalAndNonCriticalInput(currentNode);
      currentNode->removeParent(nonCritIn);
      nonCritIn->removeChild(currentNode);
      inputsY1ToYm.push_back(nonCritIn);
      currentNode = critIn;
    }

    //printAllNodesAsGraphviz(coneEnd);

    // works: printAllNodesAsGraphviz(xorStartNode);

    // XorEndNode: remove incoming edges from nodes v_1, ..., v_n
    for (auto &p : xorEndNode->getParents()) {
      // keep the operator node
      if (dynamic_cast<Operator*>(p) == nullptr) {
        p->removeChild(xorEndNode);
        xorEndNode->removeParent(p);
      }
    }

    // works: printAllNodesAsGraphviz(coneStartNodes.front());

    // XorStartNode: remove incoming edge from xorStartNode to v_t
    for (auto &c : xorStartNode->getChildren()) {
      c->removeParent(xorStartNode);
      xorStartNode->removeChild(coneEnd);
    }

    // works: printAllNodesAsGraphviz(xorStartNode);

    // TODO handle case that inputsY1ToYm.size() == 0?
    // if #xInp == 1: connect input y_1 directly to u_y
    LogicalExpr* u_y = nullptr;
    if (inputsY1ToYm.size() == 1) {
      u_y = new LogicalExpr(static_cast<AbstractExpr*>(a_t),
                            OpSymb::logicalAnd,
                            static_cast<AbstractExpr*>(inputsY1ToYm.front()));
    } else if (inputsY1ToYm.size() > 1) {
      // TODO check that this branch works
      // otherwise build XOR chain of inputs and connect last one as input of u_y
      std::vector<Node*> yXorChain = ConeRewriter::rewriteMultiInputGate(inputsY1ToYm, OpSymb::logicalXor);
      std::for_each(yXorChain.begin(), yXorChain.end(), [&](Node* n) { n->swapChildrenParents(); });
      u_y = new LogicalExpr(a_t, OpSymb::logicalAnd, yXorChain.back());
    } else {
      throw std::logic_error("Unexpected!");
    }
    finalXorInputs.push_back(u_y);

    std::cout << "-----------" << std::endl;
    // works: printAllNodesAsGraphviz(u_y);

    // for each of these start nodes v_i
    for (auto sNode : coneStartNodes) {
      auto sNodeLexp = dynamic_cast<LogicalExpr*>(sNode);
      assert(sNodeLexp != nullptr);

      // determine critical input a_1^i and non-critical input a_2^i of v_i
      auto[criticalInput, nonCriticalInput] = getCriticalAndNonCriticalInput(sNode);

      // remove critical input of v_i
      criticalInput->removeChild(sNode);
      sNode->removeParents();

      // remove all outgoing edges of v_i
      for (auto &c : sNode->getChildren()) c->removeParent(sNode);
      sNode->removeChildren();

      // add non-critical input of v_t (coneEnd) named a_t as input to v_i
      sNodeLexp->setAttributes(dynamic_cast<AbstractExpr*>(nonCriticalInput),
                               sNodeLexp->getOp(),
                               dynamic_cast<AbstractExpr*>(a_t));

      // create new logical-AND node u_i and set v_i as input of u_i
      auto leftOp = dynamic_cast<AbstractExpr*>(criticalInput);
      auto uNode = new LogicalExpr(leftOp, OpSymb::logicalAnd, sNode);
      finalXorInputs.push_back(uNode);
    }

    // TODO continue debugging here
    // TODO add "bool hasReversedEdge" as attribute to Node and method "ensureReversed()"

    //printAllNodesAsGraphviz(coneEnd);

    // convert multi-input XOR into binary XOR nodes
    std::vector<Node*> xorFinalGate = ConeRewriter::rewriteMultiInputGate(finalXorInputs, OpSymb::logicalXor);
    for (auto &gate : xorFinalGate) { gate->getUniqueNodeId(); }

    // and connect final XOR to cone output
    coneEnd->removeChild(rNode);
    rNode->removeParent(coneEnd);
    rNode->addParent(xorFinalGate.back());
    xorFinalGate.back()->addChild(rNode);

    std::cout << "----" << std::endl;
    ConeRewriter::printAllNodesAsGraphviz(rNode);

    // TODO write method similar to printConesAsGraphviz (but without stopping at first ancestor AND node) that
    //  prints the transformed graph

    std::cerr << "rewriteCones not implemented yet! Aborting..." << std::endl;
    exit(1);
  }
  // return optimizedAst;
}

void ConeRewriter::printAllNodesAsGraphviz(Node* pNode) {
  std::set<Node*> printedNodes;
  std::queue<Node*> q{{pNode}};
  while (!q.empty()) {
    auto curNode = q.front();
    q.pop();
    if (printedNodes.count(curNode) == 0) {
      std::cout << curNode->getDotFormattedString(true, "\t", false);
      for (auto &c : curNode->getChildren()) q.push(c);
      for (auto &p : curNode->getParents()) q.push(p);
      printedNodes.insert(curNode);
    }
  }
}


