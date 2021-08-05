#include <set>
#include <queue>
#include "ast_opt/utilities/ConeRewriter.h"
#include "ast_opt/ast/OperatorExpression.h"
#include "ast_opt/ast/Variable.h"
#include "ast_opt/visitor/GetAllNodesVisitor.h"

std::unique_ptr<AbstractNode> ConeRewriter::rewriteAst(std::unique_ptr<AbstractNode> &&ast_in) {

  /// set of all reducible cones
  auto delta = getReducibleCone(*ast_in);

  /// C^{AND} circuit consisting of critical AND-nodes that are connected if there is a multiplicative depth-2
  /// path in between two of those nodes in the initial circuit
  /// each cone δ ∈ Δ has an unique terminal AND node in C^{AND}
  std::vector<AbstractNode *> cAndCkt = getAndCriticalCircuit(*ast_in, delta);

  /// minimum set of reducible cones
  std::vector<AbstractNode *> deltaMin = selectCones(*ast_in, cAndCkt);

  /// Perform actual rewriting
  return rewriteCones(std::move(ast_in), delta);
}

std::vector<AbstractNode *> ConeRewriter::getReducibleCone(AbstractNode &root /* AbstractNode *v , int minDepth */) {

}

std::vector<AbstractNode *> ConeRewriter::getReducibleCone(AbstractNode *root,
                                                           AbstractNode *v,
                                                           int minDepth,
                                                           MultDepthMap multiplicativeDepths) {

  //TODO: check if we need to check if OperatorExpression (algo considers only logical expressions!)

  // return empty set if minDepth is reached
  if (computeMultDepthL(v, multiplicativeDepths)==minDepth) {
    return std::vector<AbstractNode *>();
  }

  // get predecessor (children) { p ∈ pred(v) | l(p) = l(v) - d(v) }: put them in the vector called pvec
  auto pvec = std::vector<AbstractNode *>();
  for (auto &p: *v) {
    if (computeMultDepthL(&p, multiplicativeDepths) == computeMultDepthL(v, multiplicativeDepths) - depthValue(v)) {
      pvec.push_back(&p);
    }
  }

  // return v if at least one predecessor of v is non-critical (i.e |pvev| < 2) and v is an AND-gate
  if (pvec.size() < 2 && v->toString(false) == "&&") {
    // return set consisting of start node v only
    return std::vector<AbstractNode *>{v};
  }

  // determine reducible input cones
  std::vector<std::vector<AbstractNode *>> deltaR;
  std::vector<AbstractNode *> delta;
  for (auto &p : pvec) { //TODO continue: need to fix computeMindepth
    //std::vector<AbstractNode *> intermedResult = getReducibleCone(root, p, computeMinDepth(p));
   // if (!intermedResult.empty()) deltaR.push_back(intermedResult);
  }

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

int ConeRewriter::computeMinDepth(AbstractNode *v, AbstractNode *ast, MultDepthMap map) {
  // find a non-critical input (child) node p of v
  auto isNoOperatorNode = [](AbstractNode *n) { return (dynamic_cast<Operator *>(n)==nullptr); };
  for (auto &p : *v) {
    if (!isCriticalNode(&p, ast) && isNoOperatorNode(&p)) {
      //std::cout << "Noncrit input " << p.toString(false) << std::endl;
      return computeMultDepthL(&p, map) + 1;
    }
  }
  // error (-1) if no non-critical child node
  return -1;
}

bool ConeRewriter::isCriticalNode(AbstractNode *n, AbstractNode *ast, MultDepthMap map) {
  int l = computeMultDepthL(n, map);
  int r = computeReversedMultDepthR(n, map);
  return (getMaximumMultDepth(ast) == l + r);
}

int ConeRewriter::getMultDepth(AbstractNode *n) {

  return 0;
}

int ConeRewriter::getReverseMultDepth(MultDepthMap multiplicativeDepthsReversed,
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

int ConeRewriter::computeMultDepthL(AbstractNode *n, MultDepthMap &map) {

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
    MultDepthMap m;
    uDepth = computeMultDepthL(u, m);
    if (uDepth > max) { max = uDepth; } // update maximum if necessary
  }
  auto r = max + depthValue(n);
  map.insert_or_assign(n->getUniqueNodeId(), r);
  return r;
}

int ConeRewriter::computeReversedMultDepthR(AbstractNode *n,
                                            MultDepthMap &multiplicativeDepthsReversed,
                                            AbstractNode *root) {

  //TODO: Stop at root, not just follow until no more parent
  //   if root is nullptr, just keep going!

// check if we have calculated the reverse multiplicative depth previously
  if (!multiplicativeDepthsReversed.empty()) {
    auto it = multiplicativeDepthsReversed.find(n->getUniqueNodeId());
    if (it!=multiplicativeDepthsReversed.end())
      return it->second;
  }

  AbstractNode *nextNodeToConsider;

  // if root node, the reverse multiplicative depth is 0
  if (!n->hasParent()) {
    return 0;
  } else {
    nextNodeToConsider = &n->getParent();
  }

  // otherwise compute the reverse depth
  int max = 0;
  int uDepthR;
  MultDepthMap m;
  uDepthR = computeReversedMultDepthR(nextNodeToConsider, m, nullptr);
  if (uDepthR > max) { max = uDepthR; }
  return max + depthValue(nextNodeToConsider);
}
int ConeRewriter::getMaximumMultDepth(AbstractNode *root, MultDepthMap map) {
  if (!map.empty()) {
    // do nothing
  } else {
    // compute map
    int tmp = computeMultDepthL(root, map);
  }
  // find and return max value
  return std::max_element(
      map.begin(), map.end(),
      [](const std::pair<const std::basic_string<char>, int> &a,
         const std::pair<const std::basic_string<char>, int> &b) {
        return a.second < b.second;
      })->second;
}

int getMultDepthL(MultDepthMap multiplicativeDepths, AbstractNode &n) {
  if (multiplicativeDepths.empty()) { return -1; }
  return multiplicativeDepths[n.getUniqueNodeId()];
}

