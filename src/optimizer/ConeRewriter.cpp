#include "LogicalExpr.h"
#include <set>
#include <queue>
#include "ConeRewriter.h"
#include "Return.h"
#include "Function.h"


// ------------------------------------–
// Algorithms presented in the paper
// ------------------------------------–

Ast &ConeRewriter::applyConeRewriting(Ast &ast) {
    validateCircuit(ast);
    // We need to reverse the tree's edges to be conform to the paper
    if (!ast.isReversed()) ConeRewriter::reverseEdges(ast);
    ast.printGraphviz();
    return ConeRewriter::applyMinMultDepthHeuristic(ast);
}

// TODO this method has not been tested yet
Ast &ConeRewriter::applyMinMultDepthHeuristic(Ast &ast) {
    // output circuit
    Ast *cOut = new Ast();

    // compute reducible cones set
    //ConeRewriter::getReducibleConesForEveryPossibleStartingNode(ast);  // TODO(pjattke): remove after debugging
    std::vector<Node *> deltaMin = ConeRewriter::computeReducibleCones(ast);

    while (!deltaMin.empty()) {
        // rewrite cones from deltaMin (in Ast or in C^{AND}?)
        auto rewrittenCkt = rewriteCones(deltaMin);

        // if depth of cone-rewritten circuit is smaller: assign new circuit to cOut
        if (getMaxMultDepth(rewrittenCkt) < getMaxMultDepth(*cOut)) {
            cOut = &rewrittenCkt;
        }

        // recompute reducible cones
        deltaMin = ConeRewriter::computeReducibleCones(ast);
    }

    // return multiplicative depth-optimized circuit
    return *cOut;
}

std::vector<Node *> ConeRewriter::computeReducibleCones(Ast &ast) {
    // delta: set of all reducible cones
    std::vector<Node *> delta = ConeRewriter::getReducibleCones(ast);
    std::cout << "delta: " << std::endl << "  " << delta << std::endl;

    // cAndCkt: C^{AND} circuit consisting of critical AND-nodes that are connected if there is a multiplicative depth-2
    // path in between two of those nodes in the initial circuit
    // each cone δ ∈ Δ has an unique terminal AND node in C^{AND}
    // std::vector<Node*> cAndCkt = ConeRewriter::getAndCriticalCircuit(ast); // TODO
    std::vector<Node *> cAndCkt = ConeRewriter::getAndCriticalCircuit(delta);
    std::cout << "cAndCkt:" << std::endl << "  " << cAndCkt << std::endl;

    // deltaMin: the minimum set of reducible cones
    std::vector<Node *> deltaMin = ConeRewriter::selectCones(cAndCkt);
    std::cout << "deltaMin: " << std::endl << "  " << deltaMin << std::endl;
    return deltaMin;
}

std::vector<Node *> ConeRewriter::getReducibleCones(Ast &ast) {
    Node *startNode = nullptr;
    auto rootNode = ast.getRootNode();

    auto getNthAncestorLogicalExpr = [&](int N) -> Node * {
        // find first LogicalExpr in AST
        std::pair<Node *, int> candidate(nullptr, 0);
        std::queue<std::pair<Node *, int>> q{{std::make_pair(rootNode, 0)}};
        while (!q.empty()) {
            auto curNode = q.front().first;
            auto curLevel = q.front().second;
            q.pop();
            if (dynamic_cast<LogicalExpr *>(curNode)) {
                candidate = std::make_pair(curNode, candidate.second + 1);
                if (candidate.first != nullptr && candidate.second == N)
                    return candidate.first;
            }
            std::for_each(curNode->getParents().begin(), curNode->getParents().end(), [&](Node *n) {
                q.push(std::make_pair(n, curLevel + 1));
            });
        }
        return nullptr;
    };

    // make sure that v is a LogicalExpr, otherwise find first (N=1) LogicalExpr by traversing parents
    if (dynamic_cast<LogicalExpr *>(rootNode) == nullptr) {
        startNode = getNthAncestorLogicalExpr(1);
        if (!startNode) throw std::logic_error("AST does not contain any LogicalExpr!");
    }

    // compute minDepth required by getReducibleCones algorithm
    int minDepth = computeMinDepth(startNode);

    // v has no non-critical input node p -> return empty set
    if (minDepth == -1) return std::vector<Node *>();

    std::cout << "getReducibleCones(" << startNode->getUniqueNodeId() << ", " << minDepth << ")" << std::endl;
    return getReducibleCones(startNode, minDepth);
}



// ------------------------------------–
// Internal (non-universal) helper methods
// ------------------------------------–

int ConeRewriter::computeMinDepth(Node *v) {
    // find a non-critical input node p of v
    auto isNoOperatorNode = [](Node *n) { return (dynamic_cast<Operator *>(n) == nullptr); };
    int lMax = getMultDepthL(v);
    for (auto &p : v->getPred()) {
        // exclude Operator nodes as they do not have any parent and are not modeled as node in the paper
        if (isNoOperatorNode(p) && !isCriticalNode(lMax, p)) {
            // set minMultDepth as l(p)+2 and call getReducibleCones
            return getMultDepthL(p) + 2;
            // According to the paper (see p. 9, §2.4) minMultDepth = l(p)+1 is used but it does not return any result:
            // return getMultDepthL(p) + 1;
        }
    }
    // return -1 (error) if node v has no non-critical input node
    return -1;
}

std::vector<Node *> ConeRewriter::getReducibleCones(Node *v, int minDepth) {
    // return empty set if minDepth is reached
    if (getMultDepthL(v) == minDepth) return std::vector<Node *>();

    // get critical predecessor nodes: { p ∈ pred(v) | l(p) = l(v) - d(v) }
    std::vector<Node *> *P = getCriticalPredecessors(v);

    // return v if at least one predecessor of v is non-critical and v is an AND-gate
    auto logicalExp = dynamic_cast<LogicalExpr *>(v);
    if (P->size() < 2 && logicalExp != nullptr && logicalExp->getOp().equals(OpSymb::logicalAnd)) {
        // return set consisting of start node v only
        return std::vector<Node *>{v};
    }

    // determine reducible input cones
    std::vector<std::vector<Node *>> deltaR;
    std::vector<Node *> delta;
    for (auto &p : *P) {
        std::vector<Node *> intermedResult = getReducibleCones(p, computeMinDepth(p));
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
      return std::vector<Node *>();
  }

  if (logicalExp->getOp().equals(OpSymb::logicalAnd)) {
      // both cones must be reducible because deltaR is non-empty -> pick a random one, and assign to delta
      std::cout << "> both cones must be reducible, picking randomly from sets: { ";
      for (auto &vec : deltaR) std::cout << "  " << vec;
      std::cout << " }" << std::endl;
      delta = *select_randomly(deltaR.begin(), deltaR.end());
  } else if (logicalExp->getOp().equals(OpSymb::logicalXor)) {
    // critical cones must be reducible because size of deltaR equals size of P
    // flatten vector deltaR consisting of sets generated each by getReducibleCones
      std::vector<Node *> flattenedDeltaR;
    flattenVectors(flattenedDeltaR, deltaR);
    // add all elements of flattened deltaR to delta
    addElements(delta, flattenedDeltaR);
  }

  // return 𝛅 ⋃ {v}
  delta.push_back(v);
  return delta;
}

int ConeRewriter::getMaxMultDepth(const Ast &ast) {
    std::queue<Node *> nodesToCheck;
    nodesToCheck.push(ast.getRootNode());
    int highestDepth = 0;
    while (!nodesToCheck.empty()) {
        auto curNode = nodesToCheck.front();
        nodesToCheck.pop();
        highestDepth = std::max(highestDepth, getMultDepthL(curNode));
        auto vec = ast.isReversed() ? curNode->getParents() : curNode->getChildren();
        std::for_each(vec.begin(), vec.end(), [&](Node *n) {
            nodesToCheck.push(n);
        });
    }
    return highestDepth;
}

void ConeRewriter::addElements(std::vector<Node *> &result, std::vector<Node *> newElements) {
    result.reserve(result.size() + newElements.size());
    result.insert(result.end(), newElements.begin(), newElements.end());
}

void ConeRewriter::flattenVectors(std::vector<Node *> &resultVector, std::vector<std::vector<Node *>> vectorOfVectors) {
    std::set<Node *> res;
    std::for_each(vectorOfVectors.begin(), vectorOfVectors.end(), [&](std::vector<Node *> rVec) {
        res.insert(rVec.begin(), rVec.end());
        //resultVector.insert(resultVector.end(), rVec.begin(), rVec.end());
    });
    resultVector.assign(res.begin(), res.end());
}

std::vector<Node *> *ConeRewriter::getCriticalPredecessors(Node *v) {
    // P <- { p ∈ pred(v) | l(p) = l(v) - d(v) }
    auto result = new std::vector<Node *>();
    int criterion = getMultDepthL(v) - depthValue(v);
    for (auto &p : v->getPred()) {
        if (getMultDepthL(p) == criterion) result->push_back(p);
    }
    return result;
}

void ConeRewriter::validateCircuit(Ast &ast) {
    // At the moment only circuits consisting of Function, Return and LogicalExp are supported
    std::set<std::string>
            acceptedNodeNames = {LogicalExpr().getNodeName(),
                                 Return().getNodeName(),
                                 Function().getNodeName()};
    std::queue<Node *> nodesToCheck;
    nodesToCheck.push(ast.getRootNode());
    while (!nodesToCheck.empty()) {
        auto curNode = nodesToCheck.front();
        nodesToCheck.pop();
        if (acceptedNodeNames.count(curNode->getNodeName()) == 0) {
            throw std::logic_error("Cannot apply cone rewriting on AST because it consists of non-supported nodes.");
        }
    }
}

void ConeRewriter::reverseEdges(Ast &ast) {
    std::queue<Node *> nodesToCheck;
    nodesToCheck.push(ast.getRootNode());
    while (!nodesToCheck.empty()) {
        auto curNode = nodesToCheck.front();
        nodesToCheck.pop();
        std::for_each(curNode->getChildren().begin(), curNode->getChildren().end(), [&](Node *n) {
            nodesToCheck.push(n);
        });
        curNode->swapChildrenParents();
    }
    ast.toggleIsReversed();
}

void ConeRewriter::reverseEdges(const std::vector<Node *> &nodes) {
    for (Node *n : nodes) {
        n->swapChildrenParents();
    }
}

void ConeRewriter::getReducibleConesForEveryPossibleStartingNode(Ast &ast) {
    std::deque<Node *> dq;
    dq.push_back(ast.getRootNode());
    while (!dq.empty()) {
        Node *n = dq.front();
        dq.pop_front();
        std::cout << "getReducibleCones(" << n->getUniqueNodeId() << "): " << std::endl;
        std::cout << "  " << getReducibleCones(n, computeMinDepth(n)) << std::endl << std::endl;
        auto vec = (ast.getRootNode()->getParents().empty()) ? n->getChildren() : n->getParents();
        std::for_each(vec.begin(), vec.end(), [&](Node *child) { dq.push_back(child); });
    }
}

bool ConeRewriter::isCriticalNode(int lMax, Node *n) {
    int l = getMultDepthL(n);
    int r = getReverseMultDepthR(n);
    return (lMax == l + r);
}


//std::vector<Node*> ConeRewriter::getAndCriticalCircuit(const Ast &ast) {
//  // find all critical AND nodes
//  // - determine max multiplicative depth l_{max}
//  int lMax = getMaxMultDepth(ast);
//
//  // - find nodes for which l_{max} = l(v) + r(v) holds -> critical nodes
//  std::cout << "AST_{criticalNodes}:" << std::endl << "  [";
//  std::unordered_map<std::string, Node*> criticalNodes;
//  for (auto &node : getAnc(ast.getRootNode())) {
//    auto lexp = dynamic_cast<LogicalExpr*>(node);
//    if (lexp != nullptr && lexp->getOp().equals(OpSymb::logicalAnd)) {
//      if (ConeRewriter::isCriticalNode(lMax, node)) {
//        criticalNodes.emplace(node->getUniqueNodeId(), node);
//        std::cout << node->getUniqueNodeId() << ", ";
//      }
//    }
//  }
//  std::cout << "\b\b]" << std::endl;
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

std::vector<Node *> ConeRewriter::getAndCriticalCircuit(std::vector<Node *> delta) {
    // duplicate critical nodes to create new circuit C_{AND} as we do not want to modify the original circuit
    // note that clone() does not copy the links to parents and children
    std::map<std::string, Node *> cAndMap;
    std::vector<Node *> cAndResultCkt;
    for (auto &v : delta) {
        auto clonedNode = v->clone();
        clonedNode->setUnderlyingNode(v);
        cAndMap.emplace(v->getUniqueNodeId(), clonedNode);
        cAndResultCkt.push_back(clonedNode);
    }

    if (delta.size() < 2) return cAndResultCkt;

    // check if there are depth-2 critical paths in between critical nodes in the original ckt
    for (auto &v : delta) {
        std::queue<Node *> q{{v}};
        while (!q.empty()) {
            auto curNode = q.front();
            q.pop();
            for (auto &child : curNode->getChildren()) {
                auto childLexp = dynamic_cast<LogicalExpr *>(child);
                // if the child is a LogicalExpr of type AND-gate
                if (childLexp != nullptr && childLexp->getOp().equals(OpSymb::logicalAnd)) {
                    // check if this child is a critical node, if yes: connect both nodes
                    if (std::find(delta.begin(), delta.end(), childLexp) != delta.end()) {
                        Node *copiedV = cAndMap[v->getUniqueNodeId()];
                        Node *copiedChild = cAndMap[child->getUniqueNodeId()];
                        copiedV->addChild(copiedChild);
                        copiedChild->addParent(copiedV);
                        // std::cout << "added edge: { " << copiedV->getUniqueNodeId() << " -> " << copiedChild->getUniqueNodeId()
                        // << " }" << std::endl;
                    }
                } else {  // continue if the child is not a LogicalExpr --> node does not influence the multiplicative depth
                    q.push(child);
                }
            }
        }
    }
    return cAndResultCkt;
}

/// This method implements Kahn's algorithm as described at wikipedia,
/// see https://en.wikipedia.org/wiki/Topological_sorting#CITEREFKahn1962
std::vector<Node *> ConeRewriter::sortTopologically(const std::vector<Node *> &nodes) {
    std::vector<Node *> L;
    std::map<Node *, int> numEdgesDeleted;

    // S <- nodes without an incoming edge
    std::vector<Node *> S;
    std::for_each(nodes.begin(), nodes.end(), [&](Node *n) {
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

std::vector<Node *> ConeRewriter::selectCones(std::vector<Node *> cAndCkt) {
    /// Helper function that computes the flow of all nodes in the given circuit ckt
    auto computeFlow = [](std::vector<Node *> &ckt) {
        std::map<Node *, float> computedFlow;
        std::map<std::pair<Node *, Node *>, float> edgeFlows;
        for (Node *v : ConeRewriter::sortTopologically(ckt)) {
            if (v->getParents().empty()) {  // if v is input
                // trivial flow of 1
                computedFlow[v] = 1;
            } else {  // if v is intermediate node
                // compute flow by accumulating flows of incoming edges (u,v) where u ∈ pred(v)
                auto predecessorsOfV = v->getPred();
                float flow = 0.0f;
                std::for_each(predecessorsOfV.begin(), predecessorsOfV.end(), [&](Node *u) {
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

    std::vector<Node *> deltaMin;
    while (!cAndCkt.empty()) {
        // compute all node flows f^{+}(v) in cAndCkt
        std::map<Node *, float> nodeFlows = computeFlow(cAndCkt);

        // reverse circuit edges and compute all ascending node flows f^{-}(v)
        ConeRewriter::reverseEdges(cAndCkt);
        std::map<Node *, float> nodeFlowsAscending = computeFlow(cAndCkt);

        // compute f(v) for all nodes
        std::map<Node *, float> flows;
        std::for_each(cAndCkt.begin(), cAndCkt.end(), [&](Node *n) {
            flows[n] = nodeFlowsAscending[n] * nodeFlows[n];
        });

        // find the node with the largest flow u
        Node *u = std::max_element(flows.begin(), flows.end(),
                                   [](const std::pair<Node *, float> &p1, const std::pair<Node *, float> &p2) {
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

Node *ConeRewriter::getNonCriticalLeafNode(Ast &ast) {
    Node *candidateNode = nullptr;
    const int maxDepth = getMaxMultDepth(ast);
    std::queue<Node *> q{{ast.getRootNode()}};
    while (!q.empty()) {
        auto curNode = q.front();
        q.pop();
        for (auto &p : curNode->getParents()) {
            // remember found non-critical leaf node
            if (p->getParents().empty() && maxDepth != (getMultDepthL(p) + getReverseMultDepthR(p))) {
                candidateNode = p;
            }
            // enqueue the parent nodes and continue
            q.push(p);
        }
    }
    // returns the non-critical leaf node with the highest tree-depth
    return candidateNode;
}

int ConeRewriter::depthValue(Node *n) {
    if (auto lexp = dynamic_cast<LogicalExpr *>(n)) {
        return (lexp->getOp().equals(OpSymb::logicalAnd)) ? 1 : 0;
    }
    return 0;
}

int ConeRewriter::getMultDepthL(Node *v) {
    // |pred(v)| = 0 <=> v does not have any parent node
    if (v->getPred().empty()) {
        return 0;
    }
    // otherwise return max_{u ∈ pred(v)} l(u) + d(v)
    int max = 0;
    for (auto &u : v->getPred()) {
        max = std::max(getMultDepthL(u) + depthValue(v), max);
    }
    return max;
}

int ConeRewriter::getReverseMultDepthR(Node *v) {
    if (v->getChildren().empty()) {
        return 0;
    }
    int max = 0;
    for (auto &u : v->getSucc()) {
        max = std::max(getReverseMultDepthR(u) + depthValue(u), max);
    }
    return max;
}

std::vector<Node *> ConeRewriter::getAnc(Node *n) {
    auto result = new std::set<Node *>;
    std::queue<Node *> processQueue;
    processQueue.push(n);
    while (!processQueue.empty()) {
        auto curNode = processQueue.front();
        processQueue.pop();
        auto nextNodes = curNode->getParents();
        std::for_each(nextNodes.begin(), nextNodes.end(), [&](Node *node) {
            result->insert(node);
            processQueue.push(node);
        });
    }
    return std::vector<Node *>(result->begin(), result->end());
}

Ast &ConeRewriter::rewriteCones(std::vector<Node *> coneEndNode) {
    // Assumption: coneEndNode is the node representing a cone ending at coneEndNode
    // TODO implement me!
    std::cerr << "rewriteCones not implemented yet! Aborting..." << std::endl;
    exit(1);
    return *new Ast();
}


