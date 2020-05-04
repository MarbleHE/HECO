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
  // find the largest possible isBatchable subtree
  while (!processingQ.empty()) {
    auto curNode = processingQ.front();
    processingQ.pop();
    std::cout << "curNode: " << curNode->getUniqueNodeId() << " | " << curNode->getChildrenNonNull().size()
              << std::endl;
    if (isBatchableSubtree(curNode)) {
      // returns the largest batchable subtree that was found
      return curNode;
    } else {
      // enqueue all OperatorExprs as any other will not be a batchable subtree
      for (auto c : curNode->getChildrenNonNull()) {
        if (dynamic_cast<OperatorExpr *>(c)) processingQ.push(c);
      }
    }
  }
  // nullptr = no batchable subtree found
  return nullptr;
}

bool BatchingChecker::isTransparentNode(AbstractNode *node) {
  // a node is considered as transparent if it is an OperatorExpr because it can be batched by expanding any other
  // expression using the neutral element e.g., a and b*2 –- can be batched as a*1 and b*2
  return dynamic_cast<OperatorExpr *>(node)!=nullptr;
}

std::vector<AbstractNode *> BatchingChecker::getChildren(AbstractNode *node) {
  if (isTransparentNode(node)) {
    // no deeper check required as we allow (for the sake of simplicity) only max. 1 level deep transparent node
    // children of childrens because we "skip" the transparent node
    std::vector<AbstractNode *> children;
    int numGrandChildren;
    for (auto child : node->castTo<OperatorExpr>()->getOperands()) {
      auto grandChildren = child->getChildrenNonNull();
      numGrandChildren = grandChildren.size();
      children.insert(children.end(), grandChildren.begin(), grandChildren.end());
    }
    return children;
  } else if (auto nodeAsMatrixElemRef = dynamic_cast<MatrixElementRef *>(node)) {
    // do not enqueue anything further as we do not consider the MatrixElementRef's indices
    return {};
  } else {
    auto children = node->getChildrenNonNull();
    return children;
  }
}

bool BatchingChecker::isBatchingCompatible(AbstractNode *baseNode, AbstractNode *curNode) {
  if (baseNode->getNodeType()!=curNode->getNodeType()) {
    if (!isTransparentNode(baseNode) && !isTransparentNode(curNode)) {
      // not compatible because nodes are of different type and we allow max. 1 transparent node per level
      return false;
    } else {
      // exactly one is transparent: due to baseNode.type != curNode.type both cannot be transparent at the same time
      return isTransparentNode(baseNode) || isTransparentNode(curNode);
    }
  } else {  // baseNode.type == curNode.type
    // type-specific checks
    if (auto baseNodeAsMatrixElementRef = dynamic_cast<MatrixElementRef *>(baseNode)) {
      auto baseNodeVar = dynamic_cast<Variable *>(baseNodeAsMatrixElementRef->getOperand());
      // as baseNode's type equals curNode's type, we know that curNodeAsMatrixElementRef != nullptr
      auto curNodeAsMatrixElementRef = dynamic_cast<MatrixElementRef *>(curNode);
      auto curNodeVar = dynamic_cast<Variable *>(curNodeAsMatrixElementRef->getOperand());
      return baseNodeVar!=nullptr && curNodeVar!=nullptr
          // refer to same variable
          && baseNodeVar->getIdentifier()==curNodeVar->getIdentifier();
    } else if (auto baseNodeAsOperatorExpr = dynamic_cast<OperatorExpr *>(baseNode)) {
      auto curNodeAsOperatorExpr = dynamic_cast<OperatorExpr *>(curNode);
      // same operator
      return baseNodeAsOperatorExpr->getOperator()==curNodeAsOperatorExpr->getOperator()
          // same number of operands
          && baseNodeAsOperatorExpr->getOperands().size()==curNodeAsOperatorExpr->getOperands().size();
    } else {
      // handles all types that do not require any special handling, e.g., LiteralInt, Variable
      // (it is sufficient for batching compatibility that baseNode and curNode have the same type in that case)
      return true;
    }
  }
}

bool BatchingChecker::isBatchableSubtree(AbstractNode *subtreeRoot) {
  std::vector<AbstractNode *> qReading({subtreeRoot});
  std::vector<AbstractNode *> qWriting;
  int numChildrenPerNode = -1;

  while (!qReading.empty()) {
    auto curNode = qReading.front();
    qReading.erase(qReading.begin());

    // enqueue children
    auto children = getChildren(curNode);
    if (numChildrenPerNode==-1) {
      numChildrenPerNode = children.size();
    } else if (numChildrenPerNode!=children.size()) {
      return false;
    }
    for (auto child : children) { if (dynamic_cast<Operator *>(child)==nullptr) { qWriting.push_back(child); }}

    if (qReading.empty()) {
      // compare nodes
      AbstractNode *baseNode = qWriting.front();
      for (auto nodeIt = std::next(qWriting.begin()); nodeIt!=qWriting.end(); ++nodeIt) {
        // check batching compatibility
        if (!isBatchingCompatible(baseNode, *nodeIt)) {
          // if we detected a batching-incompatibility, we can abort testing further
          return false;
        } else if (isTransparentNode(*nodeIt)) {
          // as we allow max. 1 transparent node per level, we need to make sure to compare any further transparent
          // nodes with the one we found here
          baseNode = *nodeIt;
        }
      } // end: for

      // move elements from one to another queue or assign and create new queue for qWriting
      // important: qWriting must be empty afterwards
      qReading = std::move(qWriting);

      // reset #children counter back to default value
      numChildrenPerNode = -1;
    }

  } // end: while

  // if we processed all nodes and did not abort in between due to failed batching compatibility, the node rooted
  // at subtreeRoot is considered as batchable
  return qReading.empty() && qWriting.empty();
}
