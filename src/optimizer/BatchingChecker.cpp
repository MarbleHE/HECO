#include <iostream>
#include <string>
#include <utility>
#include <unordered_set>
#include "ast_opt/optimizer/BatchingChecker.h"
#include "ast_opt/ast/MatrixElementRef.h"
#include "ast_opt/ast/MatrixAssignm.h"
#include "ast_opt/ast/Function.h"
#include "ast_opt/ast/Block.h"
#include "ast_opt/ast/OperatorExpr.h"

/*
 * Idea for batching "non-symmetric" subtreees:
 *  - Do traversal in BFS style
 *  - Collect information about a level in the AST
 *  - Before continuing with the next level, determine whether it is necessary to extend the AST to enable batchability
 *    - do not always expand an OperatorExpr, only if expanding this level is worthwhile, i.e., expansion enables
 *    batchability with enough parallel operation (e.g., at least 3 operations)
 */

// TODO take expression as input
// TODO add recursion that asks next expression for batchability
//  - stop if is batchable subtree -> no need to continue because
void BatchingChecker::determineBatchability(AbstractNode *startNode) {
  struct QueueEntry {
    AbstractNode *node;
    int treeLevel;

    QueueEntry(AbstractNode *pNode, int i) : node(pNode), treeLevel(i) {};
  };

  struct OperationInfo {
    OperatorExpr *oe;

    explicit OperationInfo(OperatorExpr *oe) : oe(oe) {};
  };

  struct OperandInfo {
    int treeLevel;
    std::string variableIdentifier;

    OperandInfo(int treeLevel, std::string variableIdentifier)
        : treeLevel(treeLevel), variableIdentifier(std::move(variableIdentifier)) {}
  };

  // for enhanced readability
  typedef int DepthLevel;

  // Data structures for collecting relevant batching information.
  // information about operations
  std::map<DepthLevel, std::vector<OperationInfo>> batchingCandidates;
  // information about operands
  std::vector<OperandInfo> operands;
  // the number of nodes per tree level
  std::map<DepthLevel, int> numNodesPerLevel;

  // queue for traversal
  std::queue<QueueEntry> bfsTraversalQ;
  bfsTraversalQ.emplace(startNode, 0);

  while (!bfsTraversalQ.empty()) {
    auto &[curNode, curLevel] = bfsTraversalQ.front();
    bfsTraversalQ.pop();

    // count the number of nodes in each level
    numNodesPerLevel[curLevel]++;
//    std::cout << curLevel << "\t" << curNode->getUniqueNodeId() << " " << std::endl;

    // only multiplications are worth for being batched
    auto curAsOpExpr = dynamic_cast<OperatorExpr *>(curNode);
    if (curAsOpExpr!=nullptr && curAsOpExpr->getOperator()->equals(MULTIPLICATION)) {
      batchingCandidates[curLevel].emplace_back(curAsOpExpr);
    }

    // logic to decide which nodes to enqueue for processing next
    if (auto curNodeAsFunc = dynamic_cast<Function *>(curNode)) {
      bfsTraversalQ.emplace(curNodeAsFunc->getBody(), curLevel + 1);
    } else if (auto curNodeAsMA = dynamic_cast<MatrixAssignm *>(curNode)) {
      // enqueue the MatrixAssignm's value field only
      bfsTraversalQ.emplace(curNodeAsMA->getValue(), curLevel + 1);
    } else if (auto curNodeAsMxElRf = dynamic_cast<MatrixElementRef *>(curNode)) {
      // check and store referenced variable
      if (auto targetVar = dynamic_cast<Variable *>(curNodeAsMxElRf->getOperand())) {
        operands.emplace_back(curLevel, targetVar->getIdentifier());
      }
    } else {
      // by default, enqueue all children of current node
      for (auto &n : curNode->getChildrenNonNull()) {
        bfsTraversalQ.emplace(n, curLevel + 1);
      }
    }
  } // end: while

  // do checks that require knowledge of all operands that are involved in a batching operation
  for (auto it = batchingCandidates.begin(); it!=batchingCandidates.end();) {
    if (it->second.size() < 3) {
      // find all batching candidates that have at least three parallel operations otherwise the batching overhead is
      // larger than the actual advantage of using batching => remove unsuitable candidate
      it = batchingCandidates.erase(it);
    } else {
      // check structure of OperatorExpr's operands:
      // 1) number of operands
      int maxNumOperands = 0;
      OperationInfo *elemWithMaxOperands;
      for (auto &elem : it->second) {
        if (elem.oe->getOperands().size() > maxNumOperands) {
          maxNumOperands = elem.oe->getOperands().size();
          elemWithMaxOperands = &elem;
        }
      }

      // 2) check type of operands
      auto operandsOfExprWithMaxOperandsAsVec = elemWithMaxOperands->oe->getOperands();
      std::unordered_set<AbstractExpr *> operandsOfExprWithMaxOperands(operandsOfExprWithMaxOperandsAsVec.begin(),
                                                                       operandsOfExprWithMaxOperandsAsVec.end());
      // Literals of same type, Variables, or MatrixElementRefs of same variable
      std::vector<std::vector<OperationInfo>::iterator> operandsReqExpansion;
      for (auto operandIt = it->second.begin(); operandIt!=it->second.end(); ++operandIt) {
        if ((*operandIt).oe->getOperands().size() < maxNumOperands) {
          // check if there are any operands that require expansion
          operandsReqExpansion.push_back(operandIt);
        } else {
          // check if type of existing operands match
          // TODO sort both
          auto curOperandsAsVec = operandIt->oe->getOperands();
          std::unordered_set<AbstractExpr *> curOperands(curOperandsAsVec.begin(), curOperandsAsVec.end());
          for (auto &op :curOperands) {
            auto result = operandsOfExprWithMaxOperands.find(op);  // FIXME does not make sense for pointers
            if (result!=operandsOfExprWithMaxOperands.end()) {
              // compare types

            } else {
              // stop after the first mismatch - it does not make sense to continue comparing because expansion can
              // only add additional operands but we cannot change the type of existing operands
              break;
            }
          }
          // if at the end, there are any unmatched types, exclude operation from batching
//          if (!curOperands.empty()) {
//            std::cout << "Removing operation from batchingCandidates due to unsuitable expression structure.";
//            operandIt = it->second.erase(operandIt);
//          }
        }
      }

      if (!operandsReqExpansion.empty()) {
        // TODO (pjattke): expand operands that are enqueued in operandsReqExpansion by correct required type.
        throw std::runtime_error("Expansion of operands currently not supported!");
      }

      ++it;
    }
  } // end: for (auto it = batchingCandidates.begin(); it!=batchingCandidates.end();)

  std::cout << "Reached end of BatchingChecker::determineBatchability." << std::endl;
//  return batchingCandidates;
}
