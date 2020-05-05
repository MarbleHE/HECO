#include "ast_opt/optimizer/BatchingChecker.h"
#include <iostream>
#include <string>
#include <utility>
#include <unordered_set>
#include "ast_opt/ast/MatrixElementRef.h"
#include "ast_opt/ast/MatrixAssignm.h"
#include "ast_opt/ast/Function.h"
#include "ast_opt/ast/Block.h"
#include "ast_opt/ast/OperatorExpr.h"

AbstractNode *BatchingChecker::getLargestBatchableSubtree(AbstractExpr *expr) {
  std::queue<AbstractNode *> processingQ({expr});
  while (!processingQ.empty()) {
    auto curNode = processingQ.front();
    processingQ.pop();
    std::cout << "Checking subtree rooted at " << curNode->getUniqueNodeId()
              << " (#children: " << curNode->getChildrenNonNull().size() << ")" << std::endl;
    // returns the largest batchable subtree that was found
    if (isBatchableSubtree(curNode)) return curNode;
    // otherwise we need to continue our search by checking all OperatorExprs on the next level
    for (auto c : curNode->getChildrenNonNull()) { if (dynamic_cast<OperatorExpr *>(c)) processingQ.push(c); }
  }
  // nullptr = no batchable subtree found
  return nullptr;
}

bool BatchingChecker::isBatchableSubtree(AbstractNode *subtreeRoot) {
  std::vector<AbstractNode *> qReading({subtreeRoot});
  std::vector<AbstractNode *> qWriting;
  int numChildrenPerNode = -1;

  while (!qReading.empty()) {
    auto curNode = qReading.front();
    qReading.erase(qReading.begin());

    // get children that need to be checked next
    auto children = getChildren(curNode);
    if (numChildrenPerNode==-1) {
      numChildrenPerNode = children.size();
    } else if (numChildrenPerNode!=children.size()) {
      // a subtree is batching incompatible if the children have a different number of operands
      return false;
    }
    // enqueue children to process next
    qWriting.insert(qWriting.end(), children.begin(), children.end()); /* NOLINT */

    // procedure that compares batching compatibility of nodes; if there are no more nodes that need to be processed
    // (qReading.empty()) and there is no next level that needs to be checked (qWriting.empty()) then there is
    // nothing to check and this subtree is considered as batching-compatible
    if (qReading.empty() && !qWriting.empty()) {
      // compare nodes pairwise
      AbstractNode *baseNode = qWriting.front();
      for (auto nodeIt = std::next(qWriting.begin()); nodeIt!=qWriting.end(); ++nodeIt) {
        // check batching compatibility
        if (!isBatchingCompatible(baseNode, *nodeIt)) {
          // if we detected a batching incompatibility, we can stop any further testing
          return false;
        } else if (isTransparentNode(*nodeIt)) {
          // as we allow max. 1 transparent node per level, we need to make sure to compare any further transparent
          // nodes with the one we found here
          baseNode = *nodeIt;
        }
      } // end: for

      // move elements from qWriting to qReading: qWriting is empty afterwards
      qReading = std::move(qWriting);

      // reset #children counter back to default value
      numChildrenPerNode = -1;
    }
  } // end: while

  // if we processed all nodes and did not abort in between due to failed batching compatibility, the node rooted
  // at subtreeRoot is considered as batchable
  return qReading.empty() && qWriting.empty();
}

bool BatchingChecker::isBatchingCompatible(AbstractNode *baseNode, AbstractNode *curNode) {
  if (baseNode->getNodeType()!=curNode->getNodeType()) {
    // return true if...
    // - exactly one of both is transparent:
    //   (A XOR B)
    //   <=> (A && !B) || (!A && B)
    //   <=> (!A != !B)
    // - one of both is a AbstractLiteral
    return (!isTransparentNode(baseNode)!=!isTransparentNode(curNode))
        || dynamic_cast<AbstractLiteral *>(baseNode)!=nullptr
        || dynamic_cast<AbstractLiteral *>(curNode)!=nullptr;
  } else {  // baseNode.type == curNode.type
    // type-specific checks
    if (auto baseNodeAsMatrixElementRef = dynamic_cast<MatrixElementRef *>(baseNode)) {
      auto baseNodeVar = dynamic_cast<Variable *>(baseNodeAsMatrixElementRef->getOperand());
      // as baseNode's type equals curNode's type, we know that curNodeAsMatrixElementRef != nullptr
      auto curNodeAsMatrixElementRef = dynamic_cast<MatrixElementRef *>(curNode);
      auto curNodeVar = dynamic_cast<Variable *>(curNodeAsMatrixElementRef->getOperand());
      if (baseNodeVar==nullptr || curNodeVar==nullptr) {
        throw std::runtime_error("MatrixElementRef unexpectedly does not refer to a Variable");
      }
      // check that both MatrixElementRefs refer to the same variable
      return baseNodeVar->getIdentifier()==curNodeVar->getIdentifier();
    } else if (auto baseNodeAsOperatorExpr = dynamic_cast<OperatorExpr *>(baseNode)) {
      auto curNodeAsOperatorExpr = dynamic_cast<OperatorExpr *>(curNode);
      // same operator
      return *baseNodeAsOperatorExpr->getOperator()==*curNodeAsOperatorExpr->getOperator()
          // same number of operands
          && baseNodeAsOperatorExpr->getOperands().size()==curNodeAsOperatorExpr->getOperands().size();
    } else {
      // handles all types that do not require any special handling, e.g., LiteralInt, Variable
      // (it is sufficient for batching compatibility that baseNode and curNode have the same type in that case)
      return true;
    }
  }
}

bool BatchingChecker::isTransparentNode(AbstractNode *node) {
  // a node is considered as transparent if it is an OperatorExpr because it can be batched by expanding any other
  // expression using the neutral element e.g., a and b*2 –- can be batched as a*1 and b*2
  return dynamic_cast<OperatorExpr *>(node)!=nullptr;
}

std::vector<AbstractNode *> BatchingChecker::getChildren(AbstractNode *node) {
  if (auto nodeAsOperatorExpr = dynamic_cast<OperatorExpr *>(node)) {
    // return the operands only as in the level before, when this OperatorExpr was added to the queue, we already
    // compared the equality of the OperatorExprs operator
    auto operands = nodeAsOperatorExpr->getOperands();
    return std::vector<AbstractNode *>(operands.begin(), operands.end());
  } else if (auto nodeAsMatrixElemRef = dynamic_cast<MatrixElementRef *>(node)) {
    // do not enqueue anything further as we do not want to consider the MatrixElementRef's indices
    return {};
  } else {
    // enqueue all children by default
    auto children = node->getChildrenNonNull();
    return children;
  }
}

bool BatchingChecker::shouldBeBatched(AbstractNode *largestBatchableSubtreeRoot) {
  auto nodesInSubtree = largestBatchableSubtreeRoot->getDescendants();
  auto numSuitableNodes = std::count_if(nodesInSubtree.begin(), nodesInSubtree.end(), [](AbstractNode *an) {
    auto oe = dynamic_cast<OperatorExpr *>(an);
    // a node is considered worthwhile for being batched if it includes a multiplication
    return oe!=nullptr && oe->getOperator()->equals(MULTIPLICATION);
  });
  // a batchable subtree is considered as wortwhile for being batched if there are at least three multiplications
  return numSuitableNodes >= 3;
}
