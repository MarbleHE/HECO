#include <set>
#include <queue>
#include "ast_opt/utilities/ConeRewriter.h"
#include "ast_opt/ast/OperatorExpression.h"
#include "ast_opt/ast/Variable.h"
#include "ast_opt/visitor/GetAllNodesVisitor.h"

std::unique_ptr<AbstractNode> ConeRewriter::rewriteAst(std::unique_ptr<AbstractNode> &&ast_in) {

  /// set of all reducible cones
  auto delta = getReducibleCones(*ast_in);

  /// C^{AND} circuit consisting of critical AND-nodes that are connected if there is a multiplicative depth-2
  /// path in between two of those nodes in the initial circuit
  /// each cone δ ∈ Δ has an unique terminal AND node in C^{AND}
  std::vector<AbstractNode *> cAndCkt = getAndCriticalCircuit(*ast_in, delta);

  /// minimum set of reducible cones
  std::vector<AbstractNode *> deltaMin = selectCones(*ast_in, cAndCkt);

  /// Perform actual rewriting
  return rewriteCones(std::move(ast_in), delta);
}

std::vector<AbstractNode *> ConeRewriter::getReducibleCones(AbstractNode &root /* AbstractNode *v , int minDepth */) {

  //TODO: CAREFUL! OLD CODE WAS DESIGNED FOR REVERSED AST!
  AbstractNode *startNode = nullptr;
  AbstractNode *rootNode = &root; // root node of AST

  // lambda fct to find n-th ancestor logical expression (we do this in order to find our start node v for Algo 1 in Aubry at al.)
  auto getNthAncestorLogicalExpr = [&](int N) -> AbstractNode * {
    // find first OperatorExpression in AST
    //TODO: Consider if changes need to be made if we allow non-logical OperatorExpression?
    std::pair<OperatorExpression *, int> candidate(nullptr, 0);
    std::queue<AbstractNode *> q{{rootNode}};
    while (!q.empty()) {
      auto curNode = q.front();
      q.pop();
      if (auto op_exp_ptr = dynamic_cast<OperatorExpression *>(curNode)) {
        candidate = std::make_pair(op_exp_ptr, candidate.second + 1);
        if (candidate.second==N)
          return op_exp_ptr;
      }
      // add all 'parent' (paper), (here: children) nodes: note this is different from the legacy version since the AST is reversed in the old version
      for (auto &n : *curNode) { q.push(&n); }
    }
    return nullptr;
  };
  // now we have the n-th 'ancestor' (paper), in our case it's actually the n-th child operator expr

  // check if v (rootNode) is a LogicalExpr, otherwise find first (N=1) LogicalExpr by its traversing 'parents' (paper), here: child
  if (dynamic_cast<OperatorExpression *>(rootNode)==nullptr) {
    startNode = getNthAncestorLogicalExpr(1); // our start node v for the Algo1 from Aubry et al. will be this node
    if (!startNode) throw std::logic_error("AST does not contain any (logical) OperatorExpr!");
  }

  // now we get the reducible cones starting from our 'first' logical expression
  // compute minDepth required by getReducibleCones algorithm
  int minDepth = -1;  //TODO: Do this statically: computeMinDepth(startNode);

  // v has no non-critical input node p -> return empty set
  if (minDepth==-1) return std::vector<AbstractNode *>();

  return getReducibleCones(root, startNode, minDepth);
}

std::vector<AbstractNode *> ConeRewriter::getReducibleCones(AbstractNode &root, AbstractNode *v, int minDepth) {
  // return empty set if minDepth is reached
  // TODO: RENABLE if (getMultDepthL(v)==minDepth) return std::vector<AbstractNode *>();

  // get predecessor nodes on critical path, i.e., { p ∈ pred(v) | l(p) = l(v) - d(v) } : here successors!
  std::vector<AbstractNode *> *P = {}; // TODO: RENABLE getPredecessorOnCriticalPath(v);

  // return v if at least one predecessor of v is non-critical and v is an AND-gate
  auto logicalExp = dynamic_cast<BinaryExpression *>(v);

  if (P->size() < 2 && logicalExp!=nullptr && logicalExp->getOperator().toString()=="AND") {
//    // return set consisting of start node v only
    return std::vector<AbstractNode *>{v};
  }

  // determine reducible input cones
  std::vector<std::vector<AbstractNode *>> deltaR;
  std::vector<AbstractNode *> delta;
  for (auto &p : *P) {
    // TODO: RENABLE  std::vector<AbstractNode *> intermedResult = getReducibleCones(root, p, computeMinDepth(p));
    // TODO: RENABLE if (!intermedResult.empty()) deltaR.push_back(intermedResult);
  }
//
//  // return empty set if either one of the following is true:
//  // a. v is not a LogicalExpr
//  // b. v is a LogicalExpr but not an AND- or XOR-gate
//  // b. v is an AND-gate and deltaR is empty
//  // c. v is a XOR-gate and size of deltaR does not equal size of P
  if (logicalExp==nullptr ||
      !(logicalExp->getOperator().toString()=="LOGICAL_AND"
          || logicalExp->getOperator().toString()=="LOGICAL_XOR") ||
      (logicalExp->getOperator().toString()=="LOGICAL_AND" && deltaR.empty()) ||
      (logicalExp->getOperator().toString()=="LOGICAL_XOR" && deltaR.size()!=P->size())) {
    return std::vector<AbstractNode *>();
  }

  if (logicalExp->getOperator().toString()=="LOGICAL_AND") {
//    // both cones must be reducible because deltaR is non-empty -> pick a random one, and assign to delta
//    delta = *select_randomly(deltaR.begin(), deltaR.end());
  } else if (logicalExp->getOperator().toString()=="LOGICAL_XOR") {
//    // critical cones must be reducible because size of deltaR equals size of P
//    // flatten vector deltaR consisting of sets generated each by getReducibleCones
    std::vector<AbstractNode *> flattenedDeltaR;
//    flattenVectors(flattenedDeltaR, deltaR);
//    // add all elements of flattened deltaR to delta
//    addElements(delta, flattenedDeltaR);
  }
//
//  // return 𝛅 ⋃ {v}
  delta.push_back(v);
  return delta;
//  return std::vector<AbstractNode *>();
}

std::vector<AbstractNode *> ConeRewriter::getAndCriticalCircuit(AbstractNode &root, std::vector<AbstractNode *> delta) {
//  // remove non-AND nodes from delta (note: delta is passed as copy-by-value) as delta may also include XOR nodes
//    delta.erase(remove_if(delta.begin(), delta.end(), [](AbstractNode *d) {
//    auto lexp = dynamic_cast<LogicalExpression *>(d);
//    return (lexp==nullptr || !lexp->getOperator()->equals(LogCompOp::LOGICAL_AND));
//  }), delta.end());
//
//  // duplicate critical nodes to create new circuit C_{AND} as we do not want to modify the original circuit
//  std::unordered_map<std::string, AbstractNode *> cAndMap;
//  std::vector<AbstractNode *> cAndResultCkt;
//  for (auto &v : delta) {
//    // note that cloneFlat() does not copy the links to parents and children
//    auto clonedNode = v->cloneFlat();
//    // a back-link to the node in the original circuit
//    underlying_nodes.insert(std::make_pair<std::string, AbstractNode *>(v->getUniqueNodeId(), &*v));
//    cAndMap.emplace(v->getUniqueNodeId(), clonedNode);
//    cAndResultCkt.push_back(clonedNode);
//  }
//
//  // in case that there are less than two nodes, we can not connect any two nodes
//  if (delta.size() < 2) return cAndResultCkt;
//
//  // check if there are depth-2 critical paths in between critical nodes in the original ckt
//  for (auto &v : delta) {
//    std::queue<AbstractNode *> q{{v}};
//    while (!q.empty()) {
//      auto curNode = q.front();
//      q.pop();
//      for (auto &child : curNode->getChildren()) {
//        auto childLexp = dynamic_cast<LogicalExpr *>(child);
//        // if the child is a LogicalExpr of type AND-gate
//        if (childLexp!=nullptr && childLexp->getOperator()->equals(LogCompOp::LOGICAL_AND)) {
//          // check if this child is a critical node, if yes: connect both nodes
//          if (std::find(delta.begin(), delta.end(), childLexp)!=delta.end()) {
//            AbstractNode *copiedV = cAndMap[v->getUniqueNodeId()];
//            AbstractNode *copiedChild = cAndMap[child->getUniqueNodeId()];
//            copiedV->addChild(copiedChild, false);
//            copiedChild->addParent(copiedV, false);
//          }
//        } else {  // continue if child is not a LogicalExpr --> node does not influence the mult. depth
//          q.push(child);
//        }
//      }
//    }
//  }
//  return cAndResultCkt;
  return {};
}

std::vector<AbstractNode *> ConeRewriter::selectCones(AbstractNode &root, std::vector<AbstractNode *> cAndCkt) {
  return {};
}

std::unique_ptr<AbstractNode> ConeRewriter::rewriteCones(std::unique_ptr<AbstractNode> &&ast,
                                                         std::vector<AbstractNode *> &coneEndNodes) {
////  // Assumption: δ ∈ coneEndNodes represents a cone that ends at node δ
//  for (auto coneEnd : coneEndNodes) {
////    // we need to get the node he underlying circuit as C^{AND} only contains a limited subset of nodes
//    coneEnd = underlying_nodes.find(coneEnd->getUniqueNodeId())->second;
////    // determine bounds of the cone
////    // -- upper bound: parent node of cone end
//    auto& rNode = coneEnd->getParent(); // this is node r in the paper [Aubry et al. Figure 2]
////
////    // U_y, the multi-input XOR, is represented as multiple XOR nodes in our tree:
////    // [v_1, ..., v_n] --> xorN --> xorN-1 ---> xorN-2 ---> ... ---> xor1 --> sNode v_t.
////    // We denote xorN as xorEndNode and xor1 as xorStartNode. We know that xorStartNode must be the first node in the
////    // cone, i.e., the first child of the cone's end node.
//    auto coneEndAsLogicalExpr = dynamic_cast<BinaryExpression *>(coneEnd);
//    auto[xorStartNode, a_t] = getCriticalAndNonCriticalInput(coneEndAsLogicalExpr);
////
////    // we "cut-off" parent edge between coneEnd and a_t in order to reconnect a_t later (but we keep a_t's children!)
//    std::unique_ptr<AbstractExpression> a_t_taken;
//    auto coneEndcast = dynamic_cast<BinaryExpression *>(coneEnd);
//    if( !isCriticalNode(&coneEndcast->getLeft())) {
//      a_t_taken = coneEndcast->takeLeft();
//    } else { a_t_taken = coneEndcast->takeRight(); }
////
////    // -- lower bound: first AND node while following critical path
////    // find the ancestors of δ that are LogicalExpr --> start nodes (v_1, ..., v_n) of cone
//    std::vector<AbstractNode *> coneStartNodes;
//    std::queue<AbstractNode *> q{{coneEnd}};
//    while (!q.empty()) {
//      auto curNode = q.front();
//      q.pop();
//      auto curNodeLexp = dynamic_cast<BinaryExpression *>(curNode);
////      // if curNode is not the cone end node and a logical-AND expression, we found one of the end nodes
////      // in that case we can stop exploring this path any further
//      if (curNode!=coneEnd && curNodeLexp!=nullptr && curNodeLexp->toString(false) == "LOGICAL_AND") {
//        coneStartNodes.push_back(curNode);
//      } else {  // otherwise we need to continue search by following the critical path
////        // add parent nodes of current nodes -> continue BFS traversal
//          for (auto &child : *curNode) {
//       //   for (auto &child : curNode->getParentsNonNull()) {
//          if (isCriticalNode(&child)) { q.push(&child); }
//        }
//      }
//    }
////
//    std::vector<AbstractNode *> finalXorInputs;
////    // It should not make a difference which of the cone start nodes we take - all of them should have the same parent.
//    auto& xorEndNode = coneStartNodes.front()->getParent();
////
//    for (auto &startNode : coneStartNodes) assert(startNode->getParent()==xorEndNode);
////
////    // check whether we need to handle non-critical inputs y_1, ..., y_m
//    if (dynamic_cast<OperatorExpression *>(&xorEndNode)->getOperator().toString() == "LOGICAL_AND" && &xorEndNode==coneEnd) {
////      // if there are no non-critical inputs y_1, ..., y_m then the cone's end and cone's start are both connected with
////      // each other, i.e y_1 is an AND node
////      // remove the edge between the start nodes and the end node
////
//      std::vector<std::unique_ptr<OperatorExpression>> v1_to_vn(coneEnd->countChildren());
//
////     // for (auto &node : coneStartNodes) {
////     // v1_to_vn.push_back(node->take());
//     for (int ii = 0; ii < coneEnd->countChildren(); ii++) {
//        auto child = dynamic_cast<OperatorExpression *>(&xorEndNode)->takeChild(ii);
//       // v1_to_vn.emplace_back(*child); // This confuses me...
//        // coneEnd->removeChild(node, true);
//        }
//    } else {
////      // collect all non-critical inputs y_1, ..., y_m in between xorStartNode up to xorEndNode
//      std::vector<AbstractNode *> inputsY1ToYm;
//      auto currentNode = xorStartNode;
//      while (true) {
//        auto *currentNodeAsLogicalExpr = dynamic_cast<BinaryExpression *>(currentNode);
//        if (currentNode==nullptr)
//          throw std::logic_error(
//              "AbstractNode between cone end and cone start node is expected to be a logical expression!");
//        auto[critIn, nonCritIn] = getCriticalAndNonCriticalInput(currentNodeAsLogicalExpr);
////TODO        currentNode->removeChild(nonCritIn, false);
////TODO        nonCritIn->removeParent(currentNode, false);
//        inputsY1ToYm.push_back(nonCritIn);
//        currentNode = critIn;
////        // termination criteria (1): we reached the end of the XOR chain
//        if (currentNode->getUniqueNodeId() == xorEndNode.getUniqueNodeId()) {
//          auto *xorEndNodeAsLogicalExpr = dynamic_cast<BinaryExpression *>(currentNode);
////          // if only one of both is critical, we need to enqueue the non-critical of them.
////          // exactly one of both inputs is non-critical
////          //   <=> left is critical XOR right is critical
////          //   <=> (left is critical) unequal (right is critical)
//          if (isCriticalNode(&xorEndNodeAsLogicalExpr->getLeft())
//              !=isCriticalNode(&xorEndNodeAsLogicalExpr->getRight())) {
////            // then we need to collect this non-critical input
//            auto nonCriticalInput = getCriticalAndNonCriticalInput(xorEndNodeAsLogicalExpr).second;
//            inputsY1ToYm.push_back(nonCriticalInput);
//          }
////          // stop traversing further as we reached the xorEndNode
//          break;
//        }
////        // termination criteria (2): we already reached a cone start node
//        if (std::find(coneStartNodes.begin(), coneStartNodes.end(), currentNode)!=coneStartNodes.end()) {
//          break;
//        }
//      }
////
////      // XorEndNode: remove incoming edges from nodes v_1, ..., v_n
////      for (auto &p : xorEndNode->getChildrenNonNull()) {
////        // keep the operator node -> only remove those nodes that are no operators
////        if (dynamic_cast<Operator *>(p)==nullptr) xorEndNode->removeChild(p, true);
////      }
////      // XorStartNode: remove incoming edge from xorStartNode to v_t
////      for (auto &c : xorStartNode->getParentsNonNull()) {
////        c->removeChild(xorStartNode, true);
////      }
////      // check that at least one y_i is present
////      if (inputsY1ToYm.empty())
////        throw std::logic_error("Unexpected number (0) of non-critical inputs y_1, ..., y_m.");
////
////      // build new node u_y that is connected to inputs y_1, ..., y_m and a_t
////      AbstractExpression *u_y_rightOperand = nullptr;
////      if (inputsY1ToYm.size()==1) {
////        // if y_1 only exists: connect input y_1 directly to u_y -> trivial case
////        u_y_rightOperand = dynamic_cast<AbstractExpression *>(inputsY1ToYm.front());
////        //u_y_rightOperand = inputsY1ToYm.front()->castTo<AbstractExpression>();
////      } else if (inputsY1ToYm.size() > 1) {  // otherwise there are inputs y_1, y_2, ..., y_m
////        // otherwise build XOR chain of inputs and connect last one as input of u_y
////        std::vector<AbstractNode *> yXorChain =
////            rewriteMultiInputGateToBinaryGatesChain(inputsY1ToYm, LOGICAL_XOR);
////        u_y_rightOperand = dynamic_cast<AbstractExpression *>(yXorChain.back());
////        //u_y_rightOperand = yXorChain.back()->castTo<AbstractExpression>();
////      }
////      auto *u_y = new BinaryExpression(a_t->castTo<AbstractExpression>(), LOGICAL_AND, u_y_rightOperand);
////      finalXorInputs.push_back(u_y);
//    } // end of handling non-critical inputs y_1, ..., y_m
////
////    // for each of these start nodes v_i
////    for (auto sNode : coneStartNodes) {
////      // determine critical input a_1^i and non-critical input a_2^i of v_i
////      auto *sNodeAsLogicalExpr = dynamic_cast<BinaryExpression *>(sNode);
////      if (sNodeAsLogicalExpr==nullptr)
////        throw std::logic_error("Start node of cone is expected to be of type LogicalExpr!");
////      auto[criticalInput, nonCriticalInput] = getCriticalAndNonCriticalInput(sNodeAsLogicalExpr);
////
////      // remove critical input of v_i
////      criticalInput->removeParent(sNode, false);
////     // sNode->removeChild(criticalInput, false);
////
////      // remove all outgoing edges of v_i
////    //  sNode->removeChild(criticalInput, true);
////
////      // add non-critical input of v_t (coneEnd) named a_t as input to v_i
////      auto originalOperator = sNodeAsLogicalExpr->getOperator();
////      sNodeAsLogicalExpr->setAttributes(dynamic_cast<BinaryExpression *>(nonCriticalInput),
////                                        originalOperator,
////                                        dynamic_cast<AbstractExpression *>(a_t));
////
////      // create new logical-AND node u_i and set v_i as input of u_i
////      auto leftOp = dynamic_cast<AbstractExpression>(criticalInput);
////      //auto leftOp = criticalInput->castTo<AbstractExpression>();
////      auto uNode = new BinaryExpression(leftOp, LOGICAL_AND, sNodeAsLogicalExpr);
////      finalXorInputs.push_back(uNode);
////    }
////    // convert multi-input XOR into binary XOR nodes
////    std::vector<AbstractNode *> xorFinalGate =
////        rewriteMultiInputGateToBinaryGatesChain(finalXorInputs, LOGICAL_XOR);
////    for (auto &gate : xorFinalGate) { gate->getUniqueNodeId(); }
////
////    // remove coneEnd
////    astToRewrite.deleteNode(&coneEnd, true);
////    assert(coneEnd==nullptr);
////
////    // connect final XOR (i.e., last LogicalExpr in the chain of XOR nodes) to cone output
////    rNode->addChild(xorFinalGate.back(), false);
////    xorFinalGate.back()->addParent(rNode, false);    // convert multi-input XOR into binary XOR nodes
////    std::vector<AbstractNode *> xorFinalGate =
////        rewriteMultiInputGateToBinaryGatesChain(finalXorInputs, LogCompOp::LOGICAL_XOR);
////    for (auto &gate : xorFinalGate) { gate->getUniqueNodeId(); }
////
////    // remove coneEnd
////    astToRewrite.deleteNode(&coneEnd, true);
////    assert(coneEnd==nullptr);
////
////    // connect final XOR (i.e., last LogicalExpr in the chain of XOR nodes) to cone output
////    rNode->addChild(xorFinalGate.back(), false);
////    xorFinalGate.back()->addParent(rNode, false);
////
//  } //end: for (auto coneEnd : coneEndNodes)
  return std::move(ast);
}

int ConeRewriter::computeMinDepth(AbstractNode *v) {
//  // Only continue if not null
//  if (v==nullptr) return 0;
//
//  // find a non-critical input node p of v
//  auto isNoOperatorNode = [](AbstractNode *n) { return (dynamic_cast<Operator *>(n)==nullptr); };
//  //for (auto &p : v->getParentsNonNull()) {
//
//  // look at all 'parents'(paper). Here: children
//  std::for_each(v->begin(), v->end(), [&](AbstractNode &p) {
////    // exclude Operator nodes as they do not have any parent and are not modeled as node in the paper
//    if (isNoOperatorNode(&p) && !isCriticalNode(&p)) {
////      // set minMultDepth as l(p)+2 and call getReducibleCones
////      //return mdc.getMultDepthL(p) + 2;
////      // According to the paper (see p. 9, §2.4) minMultDepth = l(p)+1 is used but it does not return any result:
//      return getMultDepthL(&p) + 1;
//    }
//  });
//  // return -1 (error) if node v has no non-critical input node
  return -1;
}

bool ConeRewriter::isCriticalNode(AbstractNode *n) {
  //TODO implement
  //  int l = getReverseMultDepthR(n);
  //  int r = getReverseMultDepthR(n);
  return false;
}

int ConeRewriter::getMultDepth(AbstractNode *n) {

  //  // Only continue if n is non-null
  //  if (n==nullptr) return 0;
  //
  //  // check if we have calculated the multiplicative depth previously
  //  if (!multiplicativeDepths.empty()) {
  //    auto it = multiplicativeDepths.find(n->ReUniqueNodeId());
  //    if (it!=multiplicativeDepths.end())
  //      return it->second;
  //  }
  //
  //  // next nodes to consider (children)
  //  std::vector<AbstractNode *> nextNodesToConsider;
  //  for (auto &v : *n) { nextNodesToConsider.push_back(&v); }
  //
  //  // we need to compute the multiplicative depth
  //  // trivial case: v is a leaf node, i.e., does not have any 'parent' (here: child)  node
  //  // paper: |pred(v)| = 0 => multiplicative depth = 0 (here:  |children(v)| = 0 => multdepth = 0)
  //  if (nextNodesToConsider.empty()) {
  //    multiplicativeDepths[n->getUniqueNodeId()] = 0 + getInitialDepthOrNull(n).multiplicativeDepth;
  //    return 0;
  //  }
  //
  //  // otherwise compute max_{u ∈ pred(v)} l(u) + d(v)
  //  int max = 0;
  //  for (auto &u : nextNodesToConsider) {
  //    int uDepth;
  //    // compute the multiplicative depth of parent u
  //    uDepth = getMultDepthL(u);
  //    // store the computed depth
  //    multiplicativeDepths[u->getUniqueNodeId()] = uDepth + getInitialDepthOrNull(n).multiplicativeDepth;
  //    max = std::max(uDepth + depthValue(n), max);
  //  }
  //  return max;
  return 0;
}

std::unordered_map<std::string, int> computeReverseMultDepthR(AbstractNode &ast,
                                                              std::unordered_map<std::string,
                                                                                 int> multiplicativeDepthsReversed) {
  //TODO: Check if algorithm is as simple as possible?
//  AbstractNode *nextNodeToConsider = &ast;
//
//  // we need to compute the reverse multiplicative depth
//  if (nextNodeToConsider!=nullptr) {
//    multiplicativeDepthsReversed[n->getUniqueNodeId()] = 0 + getInitialDepthOrNull(n).reverseMultiplicativeDepth;
//    return 0;
//  }
//
//  // otherwise compute the reverse depth
//  int max = 0;
//  // for (auto &u : nextNodesToConsider) {
//  int uDepthR;
//  // compute the multiplicative depth of parent u
//  uDepthR = getReverseMultDepthR(nextNodeToConsider);
//  // store the computed depth
//  multiplicativeDepthsReversed[nextNodeToConsider->getUniqueNodeId()] =
//      uDepthR + getInitialDepthOrNull(n).reverseMultiplicativeDepth;
//  max = std::max(uDepthR + depthValue(nextNodeToConsider), max);
//  //}

  return multiplicativeDepthsReversed;
}

int ConeRewriter::getReverseMultDepth(std::unordered_map<std::string, int> multiplicativeDepthsReversed,
                                      AbstractNode *n) {

  // check if we have calculated the reverse multiplicative depth previously
  if (!multiplicativeDepthsReversed.empty()) {
    auto it = multiplicativeDepthsReversed.find(n->getUniqueNodeId());
    if (it!=multiplicativeDepthsReversed.end())
      return it->second;
  } else {
    throw std::runtime_error("Map is empty!");
  }
}

int ConeRewriter::depthValue(AbstractNode *n) {
  //TODO: Consider refactoring the name to "isLogicalAnd"
  if (auto lexp = dynamic_cast<OperatorExpression *>(n)) {
    // the multiplicative depth considers logical AND nodes only
    return (lexp->getOperator().toString()=="&&");
  }
  return 0;
}

int ConeRewriter::computeMultDepthL(AbstractNode *n, std::unordered_map<std::string, int> map) {

  // Only continue if n is non-null
  if (n==nullptr) return 0;

  // check if we have calculated the multiplicative depth previously
  if (!map.empty()) {
    auto it = map.find(n->getUniqueNodeId());
    if (it!=map.end())
      return it->second;
  }

  // next nodes to consider (children)
  std::vector<AbstractNode *> nextNodesToConsider;
  for (auto &v : *n) { nextNodesToConsider.push_back(&v); }

  // we need to compute the multiplicative depth
  // base case: v is a leaf node (input node), i.e., does not have any child node
  // paper: |pred(n)| = 0 => multiplicative depth = 0 + d(n) (here:  |children(n)| = 0 => multdepth = 0 + d(n))
  if (nextNodesToConsider.empty()) {
    return 0 + depthValue(n);
  }

  // otherwise compute max_{u ∈ children(n)} l(u) + d(n)
  int max = 0;
  for (auto &u : nextNodesToConsider) {
    int uDepth;
    // compute the multiplicative depth of child u
    uDepth = computeMultDepthL(u);
    if (uDepth > max) { max = uDepth; } // update maximum if necessary
  }
  return max + depthValue(n);
}

std::unordered_map<std::string, int> preComputeMultDepthsL(AbstractNode *root) {

  std::unordered_map<std::string, int> map{};
  // put all nodes of the AST starting at root in a vector (vis.v)
  GetAllNodesVisitor vis;
  root->accept(vis);

  // calculate the multiplicative depth for each node of the AST
  for (auto &node : vis.v) {
   // map[node->getUniqueNodeId()] = computeMultDepthL(node); // TODO: fix not sure why this wont work
  }

  return map;

}

int ConeRewriter::getMultDepthL(std::unordered_map<std::string, int> multiplicativeDepths, AbstractNode &n) {
  return multiplicativeDepths[n.getUniqueNodeId()];
}

//std::vector<AbstractNode *> *ConeRewriter::getPredecessorOnCriticalPath(AbstractNode *v) {
//  // P <- { p ∈ pred(v) | l(p) = l(v) - d(v) }
//  auto result = new std::vector<AbstractNode *>();
//  int criterion = getMultDepthL(v) - depthValue(v);
//  std::for_each(v->begin(), v->end(), [&](AbstractNode &p) {
//    if (getMultDepthL(&p)==criterion) result->push_back(&p);
//  });
//  return result;
//}
//
//DepthMapEntry ConeRewriter::getInitialDepthOrNull(AbstractNode *node) {
//  auto nodeAsVar = dynamic_cast<Variable *>(node);
//  if (nodeAsVar!=nullptr && initialMultiplicativeDepths.count(nodeAsVar->getIdentifier()) > 0) {
//    return initialMultiplicativeDepths.at(nodeAsVar->getIdentifier());
//  }
//  return DepthMapEntry(0, 0);
//}
//
//void ConeRewriter::precomputeMultDepths(AbstractNode *ast) {
//  // precompute the AST's multiplicative depth and reverse multiplicative depth
//  multiplicativeDepths.clear();
//  multiplicativeDepthsReversed.clear();
//
//  GetAllNodesVisitor vis;
//  ast->accept(vis);
//
//  //std::cout<< "Allnodes: " << vis.v[0]->toString(false);
//
//  for (auto &node : vis.v) {
//    getMultDepthL(node);
//    getReverseMultDepthR(node);
//  }
//
//  // determine the AST's maximum multiplicative depth
//  maximumMultiplicativeDepth = std::max_element(
//      multiplicativeDepths.begin(), multiplicativeDepths.end(),
//      [](const std::pair<const std::basic_string<char>, int> &a,
//         const std::pair<const std::basic_string<char>, int> &b) {
//        return a.second < b.second;
//      })->second;
//}
//
//int ConeRewriter::getMaximumMultiplicativeDepth() {
//  return maximumMultiplicativeDepth;
//}
//
//DepthMapEntry::DepthMapEntry(int multiplicativeDepth, int reverseMultiplicativeDepth) : multiplicativeDepth(
//    multiplicativeDepth), reverseMultiplicativeDepth(reverseMultiplicativeDepth) {}
//
//std::pair<AbstractNode *, AbstractNode *> ConeRewriter::getCriticalAndNonCriticalInput(BinaryExpression *logicalExpr) {
//  // check which of the two inputs is critical
//  bool leftIsCritical = isCriticalNode(&logicalExpr->getLeft());
//  bool rightIsCritical = isCriticalNode(&logicalExpr->getRight());
//
//  // throw exceptions if both are critical or neither of both is critical
//  if (leftIsCritical && rightIsCritical) {
//    throw std::invalid_argument("Cannot rewrite given AST because input of cone's end node are both critical!");
//  } else if (!leftIsCritical && !rightIsCritical) {
//    throw std::invalid_argument("Neither input left nor input right are critical nodes!");
//  }
//
//  // return a pair of <critical input, non-critical input>
//  return (leftIsCritical ? std::make_pair(&logicalExpr->getLeft(), &logicalExpr->getRight())
//                         : std::make_pair(&logicalExpr->getRight(), &logicalExpr->getLeft()));
//}
//
//bool ConeRewriter::isCriticalNode(AbstractNode *n) {
//  int l = mdc.getMultDepthL(n);
//  int r = mdc.getReverseMultDepthR(n);
//  return (mdc.getMaximumMultiplicativeDepth()==l + r);
//}
//
//std::pair<AbstractNode *, AbstractNode *> ConeRewriter::getCriticalAndNonCriticalInput(BinaryExpression *logicalExpr) {
//  // check which of the two inputs is critical
//  bool leftIsCritical = isCriticalNode(&logicalExpr->getLeft());
//  bool rightIsCritical = isCriticalNode(&logicalExpr->getRight());
//
//  // throw exceptions if both are critical or neither of both is critical
//  if (leftIsCritical && rightIsCritical) {
//    throw std::invalid_argument("Cannot rewrite given AST because input of cone's end node are both critical!");
//  } else if (!leftIsCritical && !rightIsCritical) {
//    throw std::invalid_argument("Neither input left nor input right are critical nodes!");
//  }
//
//  // return a pair of <critical input, non-critical input>
//  return (leftIsCritical ? std::make_pair(&logicalExpr->getLeft(), &logicalExpr->getRight())
//                         : std::make_pair(&logicalExpr->getRight(), &logicalExpr->getLeft()));
//}