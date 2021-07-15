#include <set>
#include <queue>
#include <utility>
#include "ast_opt/utilities/ConeRewriter.h"
#include "ast_opt/ast/BinaryExpression.h"
#include "ast_opt/ast/AbstractExpression.h"

ConeRewriter::ConeRewriter(AbstractNode *ast, MultiplicativeDepthCalculator mdc)
    : ast(ast), mdc(mdc) {
}



void ConeRewriter::rewriteCones(AbstractNode &astToRewrite, const std::vector<AbstractNode *> &coneEndNodes) {
  // Assumption: δ ∈ coneEndNodes represents a cone that ends at node δ
  for (auto coneEnd : coneEndNodes) {
    // we need to get the node in the underlying circuit as C^{AND} only contains a limited subset of nodes
    coneEnd = underlying_nodes.find(coneEnd->getUniqueNodeId())->second;
    // determine bounds of the cone
    // -- upper bound: parent node of cone end
    auto rNode = coneEnd->getParent();

    // U_y, the multi-input XOR, is represented as multiple XOR nodes in our tree:
    // coneEndNodes [v_1, ..., v_n] --> xorN --> xorN-1 ---> xorN-2 ---> ... ---> xor1 --> sNode v_t.
    // We denote xorN as xorEndNode and xor1 as xorStartNode. We know that xorStartNode must be the first node in the
    // cone, i.e., the first child of the cone's end node.
    auto coneEndAsLogicalExpr = dynamic_cast<BinaryExpression *>(coneEnd);
    auto[xorStartNode, a_t] = getCriticalAndNonCriticalInput(coneEndAsLogicalExpr);
    // the old algo cuts off some ties here but we skip

    // -- lower bound: first AND node while following critical path
    // find the ancestors of δ that are LogicalExpr --> start nodes (v_1, ..., v_n) of cone
    std::vector<AbstractNode *> coneStartNodes;
    std::queue<AbstractNode *> q{{coneEnd}};
    while (!q.empty()) {
      auto curNode = q.front();
      q.pop();
      auto curNodeLexp = dynamic_cast<BinaryExpression *>(curNode);
      // if curNode is not the cone end node and a logical-AND expression, we found one of the end nodes
      // in that case we can stop exploring this path any further
      if (curNode!=coneEnd && curNodeLexp!=nullptr && curNodeLexp->toString(false) == "LOGICAL_AND") {
        coneStartNodes.push_back(curNode);
      } else {  // otherwise we need to continue search by following the critical path
        // add parent nodes of current nodes -> continue BFS traversal
        for (auto &child : curNode->getChildrenNonNull()) { // TODO: HOW
//        for (auto &child : curNode->getParentsNonNull()) {
          if (isCriticalNode(child)) { q.push(child); }
        }
      }
    }

    std::vector<AbstractNode *> finalXorInputs;
    // It should not make a difference which of the cone start nodes we take - all of them should have the same parent.
    auto xorEndNode = coneStartNodes.front()->getParent();

    for (auto &startNode : coneStartNodes) assert(startNode->getParent()==xorEndNode);

    // check whether we need to handle non-critical inputs y_1, ..., y_m
    if (dynamic_cast<BinaryExpression *>(xorEndNode)->getOperator()->equals(LOGICAL_AND) && xorEndNode==coneEnd) {
      // if there are no non-critical inputs y_1, ..., y_m then the cone's end and cone's start are both connected with
      // each other.
      // remove the edge between the start nodes and the end node
      for (auto &node : coneStartNodes) {
        //  coneEnd->removeChild(node, true); // WE skip this
      }
    } else {
      // collect all non-critical inputs y_1, ..., y_m in between xorStartNode up to xorEndNode
      std::vector<AbstractNode *> inputsY1ToYm;
      auto currentNode = xorStartNode;
      while (true) {
        auto *currentNodeAsLogicalExpr = dynamic_cast<BinaryExpression *>(currentNode);
        if (currentNode==nullptr)
          throw std::logic_error(
              "AbstractNode between cone end and cone start node is expected to be a logical expression!");
        auto[critIn, nonCritIn] = getCriticalAndNonCriticalInput(currentNodeAsLogicalExpr);
        currentNode->removeChild(nonCritIn, false);
        nonCritIn->removeParent(currentNode, false);
        inputsY1ToYm.push_back(nonCritIn);
        currentNode = critIn;
        // termination criteria (1): we reached the end of the XOR chain
        if (currentNode==xorEndNode) {
          auto *xorEndNodeAsLogicalExpr = dynamic_cast<BinaryExpression *>(currentNode);
          // if only one of both is critical, we need to enqueue the non-critical of them.
          // exactly one of both inputs is non-critical
          //   <=> left is critical XOR right is critical
          //   <=> (left is critical) unequal (right is critical)
          if (isCriticalNode(xorEndNodeAsLogicalExpr->getLeft())
              !=isCriticalNode(xorEndNodeAsLogicalExpr->getRight())) {
            // then we need to collect this non-critical input
            auto nonCriticalInput = getCriticalAndNonCriticalInput(xorEndNodeAsLogicalExpr).second;
            inputsY1ToYm.push_back(nonCriticalInput);
          }
          // stop traversing further as we reached the xorEndNode
          break;
        }
        // termination criteria (2): we already reached a cone start node
        if (std::find(coneStartNodes.begin(), coneStartNodes.end(), currentNode)!=coneStartNodes.end()) {
          break;
        }
      }

      // XorEndNode: remove incoming edges from nodes v_1, ..., v_n
      for (auto &p : xorEndNode->getChildrenNonNull()) {
        // keep the operator node -> only remove those nodes that are no operators
        if (dynamic_cast<Operator *>(p)==nullptr) xorEndNode->removeChild(p, true);
      }
      // XorStartNode: remove incoming edge from xorStartNode to v_t
      for (auto &c : xorStartNode->getParentsNonNull()) {
        c->removeChild(xorStartNode, true);
      }
      // check that at least one y_i is present
      if (inputsY1ToYm.empty())
        throw std::logic_error("Unexpected number (0) of non-critical inputs y_1, ..., y_m.");

      // build new node u_y that is connected to inputs y_1, ..., y_m and a_t
      AbstractExpression *u_y_rightOperand = nullptr;
      if (inputsY1ToYm.size()==1) {
        // if y_1 only exists: connect input y_1 directly to u_y -> trivial case
        u_y_rightOperand = dynamic_cast<AbstractExpression *>(inputsY1ToYm.front());
        //u_y_rightOperand = inputsY1ToYm.front()->castTo<AbstractExpression>();
      } else if (inputsY1ToYm.size() > 1) {  // otherwise there are inputs y_1, y_2, ..., y_m
        // otherwise build XOR chain of inputs and connect last one as input of u_y
        std::vector<AbstractNode *> yXorChain =
            rewriteMultiInputGateToBinaryGatesChain(inputsY1ToYm, LOGICAL_XOR);
        u_y_rightOperand = dynamic_cast<AbstractExpression *>(yXorChain.back());
        //u_y_rightOperand = yXorChain.back()->castTo<AbstractExpression>();
      }
      auto *u_y = new BinaryExpression(a_t->castTo<AbstractExpression>(), LOGICAL_AND, u_y_rightOperand);
      finalXorInputs.push_back(u_y);
    } // end of handling non-critical inputs y_1, ..., y_m

    // for each of these start nodes v_i
    for (auto sNode : coneStartNodes) {
      // determine critical input a_1^i and non-critical input a_2^i of v_i
      auto *sNodeAsLogicalExpr = dynamic_cast<BinaryExpression *>(sNode);
      if (sNodeAsLogicalExpr==nullptr)
        throw std::logic_error("Start node of cone is expected to be of type LogicalExpr!");
      auto[criticalInput, nonCriticalInput] = getCriticalAndNonCriticalInput(sNodeAsLogicalExpr);

      // remove critical input of v_i
      criticalInput->removeParent(sNode, false);
     // sNode->removeChild(criticalInput, false);

      // remove all outgoing edges of v_i
    //  sNode->removeChild(criticalInput, true);

      // add non-critical input of v_t (coneEnd) named a_t as input to v_i
      auto originalOperator = sNodeAsLogicalExpr->getOperator();
      sNodeAsLogicalExpr->setAttributes(dynamic_cast<BinaryExpression *>(nonCriticalInput),
                                        originalOperator,
                                        dynamic_cast<AbstractExpression *>(a_t));

      // create new logical-AND node u_i and set v_i as input of u_i
      auto leftOp = dynamic_cast<AbstractExpression>(criticalInput);
      //auto leftOp = criticalInput->castTo<AbstractExpression>();
      auto uNode = new BinaryExpression(leftOp, LOGICAL_AND, sNodeAsLogicalExpr);
      finalXorInputs.push_back(uNode);
    }
    // convert multi-input XOR into binary XOR nodes
    std::vector<AbstractNode *> xorFinalGate =
        rewriteMultiInputGateToBinaryGatesChain(finalXorInputs, LOGICAL_XOR);
    for (auto &gate : xorFinalGate) { gate->getUniqueNodeId(); }

    // remove coneEnd
    astToRewrite.deleteNode(&coneEnd, true);
    assert(coneEnd==nullptr);

    // connect final XOR (i.e., last LogicalExpr in the chain of XOR nodes) to cone output
    rNode->addChild(xorFinalGate.back(), false);
    xorFinalGate.back()->addParent(rNode, false);    // convert multi-input XOR into binary XOR nodes
    std::vector<AbstractNode *> xorFinalGate =
        rewriteMultiInputGateToBinaryGatesChain(finalXorInputs, LogCompOp::LOGICAL_XOR);
    for (auto &gate : xorFinalGate) { gate->getUniqueNodeId(); }

    // remove coneEnd
    astToRewrite.deleteNode(&coneEnd, true);
    assert(coneEnd==nullptr);

    // connect final XOR (i.e., last LogicalExpr in the chain of XOR nodes) to cone output
    rNode->addChild(xorFinalGate.back(), false);
    xorFinalGate.back()->addParent(rNode, false);

  } //end: for (auto coneEnd : coneEndNodes)
}

std::pair<AbstractNode *, AbstractNode *> ConeRewriter::getCriticalAndNonCriticalInput(BinaryExpression *logicalExpr) {
  // check which of the two inputs is critical
  bool leftIsCritical = isCriticalNode(logicalExpr->getLeft());
  bool rightIsCritical = isCriticalNode(logicalExpr->getRight());

  // throw exceptions if both are critical or neither of both is critical
  if (leftIsCritical && rightIsCritical) {
    throw std::invalid_argument("Cannot rewrite given AST because input of cone's end node are both critical!");
  } else if (!leftIsCritical && !rightIsCritical) {
    throw std::invalid_argument("Neither input left nor input right are critical nodes!");
  }

  // return a pair of <critical input, non-critical input>
  return (leftIsCritical ? std::make_pair(logicalExpr->getLeft(), logicalExpr->getRight())
                         : std::make_pair(logicalExpr->getRight(), logicalExpr->getLeft()));
}

bool ConeRewriter::isCriticalNode(AbstractNode *n) {
  int l = mdc.getMultDepthL(n);
  int r = mdc.getReverseMultDepthR(n);
  return (mdc.getMaximumMultiplicativeDepth()==l + r);
}