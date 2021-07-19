#include <set>
#include <queue>
#include <utility>
#include <ast_opt/ast/OperatorExpression.h>
#include "ast_opt/utilities/ConeRewriter.h"
#include "ast_opt/ast/BinaryExpression.h"
#include "ast_opt/ast/AbstractExpression.h"

std::unique_ptr<AbstractNode> ConeRewriter::rewriteAst(std::unique_ptr<AbstractNode> &&ast_in) {
  ast = std::move(ast_in);


  //TODO: Implement getReducibleCones()
  /// set of all reducible cones
  auto delta = getReducibleCones();

  //TODO: Further processing?

  //  // cAndCkt: C^{AND} circuit consisting of critical AND-nodes that are connected if there is a multiplicative depth-2
  //  // path in between two of those nodes in the initial circuit
  //  // each cone δ ∈ Δ has an unique terminal AND node in C^{AND}
  //  std::vector<AbstractNode *> cAndCkt = ConeRewriter::getAndCriticalCircuit(delta);
  //
  //  // deltaMin: the minimum set of reducible cones
  //  std::vector<AbstractNode *> deltaMin = ConeRewriter::selectCones(cAndCkt);


  // TODO: Implement rewriteCones()
  auto new_ast = rewriteCones(delta);

  return std::move(new_ast);
}

std::unique_ptr<AbstractNode> ConeRewriter::rewriteCones(std::vector<AbstractNode *> &coneEndNodes) {
//  // Assumption: δ ∈ coneEndNodes represents a cone that ends at node δ
//  for (auto coneEnd : coneEndNodes) {
//    // we need to get the node he underlying circuit as C^{AND} only contains a limited subset of nodes
//    coneEnd = underlying_nodes.find(coneEnd->getUniqueNodeId())->second;
//    // determine bounds of the cone
//    // -- upper bound: parent node of cone end
//    auto& rNode = coneEnd->getParent(); // this is node r in the paper [Aubry et al. Figure 2]
//
//    // U_y, the multi-input XOR, is represented as multiple XOR nodes in our tree:
//    // [v_1, ..., v_n] --> xorN --> xorN-1 ---> xorN-2 ---> ... ---> xor1 --> sNode v_t.
//    // We denote xorN as xorEndNode and xor1 as xorStartNode. We know that xorStartNode must be the first node in the
//    // cone, i.e., the first child of the cone's end node.
//    auto coneEndAsLogicalExpr = dynamic_cast<BinaryExpression *>(coneEnd);
//    auto[xorStartNode, a_t] = getCriticalAndNonCriticalInput(coneEndAsLogicalExpr);
//
//    // we "cut-off" parent edge between coneEnd and a_t in order to reconnect a_t later (but we keep a_t's children!)
//    std::unique_ptr<AbstractExpression> a_t_taken;
//    auto coneEndcast = dynamic_cast<BinaryExpression *>(coneEnd);
//    if( !isCriticalNode(&coneEndcast->getLeft())) {
//      a_t_taken = coneEndcast->takeLeft();
//    } else { a_t_taken = coneEndcast->takeRight(); }
//
//    // -- lower bound: first AND node while following critical path
//    // find the ancestors of δ that are LogicalExpr --> start nodes (v_1, ..., v_n) of cone
//    std::vector<AbstractNode *> coneStartNodes;
//    std::queue<AbstractNode *> q{{coneEnd}};
//    while (!q.empty()) {
//      auto curNode = q.front();
//      q.pop();
//      auto curNodeLexp = dynamic_cast<BinaryExpression *>(curNode);
//      // if curNode is not the cone end node and a logical-AND expression, we found one of the end nodes
//      // in that case we can stop exploring this path any further
//      if (curNode!=coneEnd && curNodeLexp!=nullptr && curNodeLexp->toString(false) == "LOGICAL_AND") {
//        coneStartNodes.push_back(curNode);
//      } else {  // otherwise we need to continue search by following the critical path
//        // add parent nodes of current nodes -> continue BFS traversal
//          for (auto &child : *curNode) {
////        for (auto &child : curNode->getParentsNonNull()) {
//          if (isCriticalNode(&child)) { q.push(&child); }
//        }
//      }
//    }
//
//    std::vector<AbstractNode *> finalXorInputs;
//    // It should not make a difference which of the cone start nodes we take - all of them should have the same parent.
//    auto& xorEndNode = coneStartNodes.front()->getParent();
//
//    for (auto &startNode : coneStartNodes) assert(startNode->getParent()==xorEndNode);
//
//    // check whether we need to handle non-critical inputs y_1, ..., y_m
//    if (dynamic_cast<OperatorExpression *>(&xorEndNode)->getOperator().toString() == "LOGICAL_AND" && &xorEndNode==coneEnd) {
//      // if there are no non-critical inputs y_1, ..., y_m then the cone's end and cone's start are both connected with
//      // each other, i.e y_1 is an AND node
//      // remove the edge between the start nodes and the end node
//
//      std::vector<std::unique_ptr<OperatorExpression>> v1_to_vn(coneEnd->countChildren());
//
//     // for (auto &node : coneStartNodes) {
//     // v1_to_vn.push_back(node->take());
//     for (int ii = 0; ii < coneEnd->countChildren(); ii++) {
//        auto child = dynamic_cast<OperatorExpression *>(&xorEndNode)->takeChild(ii);
//        v1_to_vn.push_back(child); // This confuses me...
//        // coneEnd->removeChild(node, true);
//      }
//    } else {
//      // collect all non-critical inputs y_1, ..., y_m in between xorStartNode up to xorEndNode
//      std::vector<AbstractNode *> inputsY1ToYm;
//      auto currentNode = xorStartNode;
//      while (true) {
//        auto *currentNodeAsLogicalExpr = dynamic_cast<BinaryExpression *>(currentNode);
//        if (currentNode==nullptr)
//          throw std::logic_error(
//              "AbstractNode between cone end and cone start node is expected to be a logical expression!");
//        auto[critIn, nonCritIn] = getCriticalAndNonCriticalInput(currentNodeAsLogicalExpr);
//        currentNode->removeChild(nonCritIn, false);
//        nonCritIn->removeParent(currentNode, false);
//        inputsY1ToYm.push_back(nonCritIn);
//        currentNode = critIn;
//        // termination criteria (1): we reached the end of the XOR chain
//        if (currentNode==xorEndNode) {
//          auto *xorEndNodeAsLogicalExpr = dynamic_cast<BinaryExpression *>(currentNode);
//          // if only one of both is critical, we need to enqueue the non-critical of them.
//          // exactly one of both inputs is non-critical
//          //   <=> left is critical XOR right is critical
//          //   <=> (left is critical) unequal (right is critical)
//          if (isCriticalNode(xorEndNodeAsLogicalExpr->getLeft())
//              !=isCriticalNode(xorEndNodeAsLogicalExpr->getRight())) {
//            // then we need to collect this non-critical input
//            auto nonCriticalInput = getCriticalAndNonCriticalInput(xorEndNodeAsLogicalExpr).second;
//            inputsY1ToYm.push_back(nonCriticalInput);
//          }
//          // stop traversing further as we reached the xorEndNode
//          break;
//        }
//        // termination criteria (2): we already reached a cone start node
//        if (std::find(coneStartNodes.begin(), coneStartNodes.end(), currentNode)!=coneStartNodes.end()) {
//          break;
//        }
//      }
//
//      // XorEndNode: remove incoming edges from nodes v_1, ..., v_n
//      for (auto &p : xorEndNode->getChildrenNonNull()) {
//        // keep the operator node -> only remove those nodes that are no operators
//        if (dynamic_cast<Operator *>(p)==nullptr) xorEndNode->removeChild(p, true);
//      }
//      // XorStartNode: remove incoming edge from xorStartNode to v_t
//      for (auto &c : xorStartNode->getParentsNonNull()) {
//        c->removeChild(xorStartNode, true);
//      }
//      // check that at least one y_i is present
//      if (inputsY1ToYm.empty())
//        throw std::logic_error("Unexpected number (0) of non-critical inputs y_1, ..., y_m.");
//
//      // build new node u_y that is connected to inputs y_1, ..., y_m and a_t
//      AbstractExpression *u_y_rightOperand = nullptr;
//      if (inputsY1ToYm.size()==1) {
//        // if y_1 only exists: connect input y_1 directly to u_y -> trivial case
//        u_y_rightOperand = dynamic_cast<AbstractExpression *>(inputsY1ToYm.front());
//        //u_y_rightOperand = inputsY1ToYm.front()->castTo<AbstractExpression>();
//      } else if (inputsY1ToYm.size() > 1) {  // otherwise there are inputs y_1, y_2, ..., y_m
//        // otherwise build XOR chain of inputs and connect last one as input of u_y
//        std::vector<AbstractNode *> yXorChain =
//            rewriteMultiInputGateToBinaryGatesChain(inputsY1ToYm, LOGICAL_XOR);
//        u_y_rightOperand = dynamic_cast<AbstractExpression *>(yXorChain.back());
//        //u_y_rightOperand = yXorChain.back()->castTo<AbstractExpression>();
//      }
//      auto *u_y = new BinaryExpression(a_t->castTo<AbstractExpression>(), LOGICAL_AND, u_y_rightOperand);
//      finalXorInputs.push_back(u_y);
//    } // end of handling non-critical inputs y_1, ..., y_m
//
//    // for each of these start nodes v_i
//    for (auto sNode : coneStartNodes) {
//      // determine critical input a_1^i and non-critical input a_2^i of v_i
//      auto *sNodeAsLogicalExpr = dynamic_cast<BinaryExpression *>(sNode);
//      if (sNodeAsLogicalExpr==nullptr)
//        throw std::logic_error("Start node of cone is expected to be of type LogicalExpr!");
//      auto[criticalInput, nonCriticalInput] = getCriticalAndNonCriticalInput(sNodeAsLogicalExpr);
//
//      // remove critical input of v_i
//      criticalInput->removeParent(sNode, false);
//     // sNode->removeChild(criticalInput, false);
//
//      // remove all outgoing edges of v_i
//    //  sNode->removeChild(criticalInput, true);
//
//      // add non-critical input of v_t (coneEnd) named a_t as input to v_i
//      auto originalOperator = sNodeAsLogicalExpr->getOperator();
//      sNodeAsLogicalExpr->setAttributes(dynamic_cast<BinaryExpression *>(nonCriticalInput),
//                                        originalOperator,
//                                        dynamic_cast<AbstractExpression *>(a_t));
//
//      // create new logical-AND node u_i and set v_i as input of u_i
//      auto leftOp = dynamic_cast<AbstractExpression>(criticalInput);
//      //auto leftOp = criticalInput->castTo<AbstractExpression>();
//      auto uNode = new BinaryExpression(leftOp, LOGICAL_AND, sNodeAsLogicalExpr);
//      finalXorInputs.push_back(uNode);
//    }
//    // convert multi-input XOR into binary XOR nodes
//    std::vector<AbstractNode *> xorFinalGate =
//        rewriteMultiInputGateToBinaryGatesChain(finalXorInputs, LOGICAL_XOR);
//    for (auto &gate : xorFinalGate) { gate->getUniqueNodeId(); }
//
//    // remove coneEnd
//    astToRewrite.deleteNode(&coneEnd, true);
//    assert(coneEnd==nullptr);
//
//    // connect final XOR (i.e., last LogicalExpr in the chain of XOR nodes) to cone output
//    rNode->addChild(xorFinalGate.back(), false);
//    xorFinalGate.back()->addParent(rNode, false);    // convert multi-input XOR into binary XOR nodes
//    std::vector<AbstractNode *> xorFinalGate =
//        rewriteMultiInputGateToBinaryGatesChain(finalXorInputs, LogCompOp::LOGICAL_XOR);
//    for (auto &gate : xorFinalGate) { gate->getUniqueNodeId(); }
//
//    // remove coneEnd
//    astToRewrite.deleteNode(&coneEnd, true);
//    assert(coneEnd==nullptr);
//
//    // connect final XOR (i.e., last LogicalExpr in the chain of XOR nodes) to cone output
//    rNode->addChild(xorFinalGate.back(), false);
//    xorFinalGate.back()->addParent(rNode, false);
//
//  } //end: for (auto coneEnd : coneEndNodes)
  return nullptr;
}

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

std::vector<AbstractNode *> ConeRewriter::getReducibleCones(/* AbstractNode *v , int minDepth */) {

  //TODO: CAREFUL! OLD CODE WAS DESIGNED FOR REVERSED AST!
  AbstractNode *startNode = nullptr;
  AbstractNode *rootNode = ast.get(); //TODO: WAS ROOT NODE THE SAME IN REVERSED AST?

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
      // add all parent nodes //TODO: WHAT IS A PARENT HERE?
      // std::for_each(curNode->getParents().begin(), curNode->getParents().end(), [&](AbstractNode *n) { q.push(n); });
    }
    return nullptr;
  };

  // check if v (rootNode) is a LogicalExpr, otherwise find first (N=1) LogicalExpr by its traversing parents
  if (dynamic_cast<OperatorExpression *>(rootNode)==nullptr) {
    startNode = getNthAncestorLogicalExpr(1);
    if (!startNode) throw std::logic_error("AST does not contain any LogicalExpr!");
  }
//
//  // compute minDepth required by getReducibleCones algorithm
//  int minDepth = computeMinDepth(startNode);
//
//  // v has no non-critical input node p -> return empty set
//  if (minDepth==-1) return std::vector<AbstractNode *>();
//
//  return getReducibleCones(startNode, minDepth);



//  // return empty set if minDepth is reached
//  if (mdc.getMultDepthL(v)==minDepth) return std::vector<AbstractNode *>();
//
//  // get predecessor nodes on critical path, i.e., { p ∈ pred(v) | l(p) = l(v) - d(v) }
//  std::vector<AbstractNode *> *P = getPredecessorOnCriticalPath(v);
//
//  // return v if at least one predecessor of v is non-critical and v is an AND-gate
//  auto logicalExp = v->castTo<LogicalExpr>();
//  if (P->size() < 2 && logicalExp!=nullptr && logicalExp->getOperator()->equals(LogCompOp::LOGICAL_AND)) {
//    // return set consisting of start node v only
//    return std::vector<AbstractNode *>{v};
//  }
//
//  // determine reducible input cones
//  std::vector<std::vector<AbstractNode *>> deltaR;
//  std::vector<AbstractNode *> delta;
//  for (auto &p : *P) {
//    std::vector<AbstractNode *> intermedResult = getReducibleCones(p, computeMinDepth(p));
//    if (!intermedResult.empty()) deltaR.push_back(intermedResult);
//  }
//
//  // return empty set if either one of the following is true:
//  // a. v is not a LogicalExpr
//  // b. v is a LogicalExpr but not an AND- or XOR-gate
//  // b. v is an AND-gate and deltaR is empty
//  // c. v is a XOR-gate and size of deltaR does not equal size of P
//  if (logicalExp==nullptr ||
//      !(logicalExp->getOperator()->equals(LogCompOp::LOGICAL_AND)
//          || logicalExp->getOperator()->equals(LogCompOp::LOGICAL_XOR)) ||
//      (logicalExp->getOperator()->equals(LogCompOp::LOGICAL_AND) && deltaR.empty()) ||
//      (logicalExp->getOperator()->equals(LogCompOp::LOGICAL_XOR) && deltaR.size()!=P->size())) {
//    return std::vector<AbstractNode *>();
//  }
//
//  if (logicalExp->getOperator()->equals(LogCompOp::LOGICAL_AND)) {
//    // both cones must be reducible because deltaR is non-empty -> pick a random one, and assign to delta
//    delta = *select_randomly(deltaR.begin(), deltaR.end());
//  } else if (logicalExp->getOperator()->equals(LogCompOp::LOGICAL_XOR)) {
//    // critical cones must be reducible because size of deltaR equals size of P
//    // flatten vector deltaR consisting of sets generated each by getReducibleCones
//    std::vector<AbstractNode *> flattenedDeltaR;
//    flattenVectors(flattenedDeltaR, deltaR);
//    // add all elements of flattened deltaR to delta
//    addElements(delta, flattenedDeltaR);
//  }
//
//  // return 𝛅 ⋃ {v}
//  delta.push_back(v);
//  return delta;
  return std::vector<AbstractNode *>();
}

//std::vector<AbstractNode *> ConeRewriter::getReducibleCones() {
//
//}
//
//std::vector<AbstractNode *> ConeRewriter::computeReducibleCones() {
//
//}
//
//std::vector<AbstractNode *> *ConeRewriter::getPredecessorOnCriticalPath(AbstractNode *v) {
//  // P <- { p ∈ pred(v) | l(p) = l(v) - d(v) }
//  auto result = new std::vector<AbstractNode *>();
//  int criterion = mdc.getMultDepthL(v) - mdc.depthValue(v);
//  for (auto &p : v->getParentsNonNull()) {
//    if (mdc.getMultDepthL(p)==criterion) result->push_back(p);
//  }
//  return result;
//}
//
//
//
