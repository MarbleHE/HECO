#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_OPTIMIZER_BATCHINGCHECKER_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_OPTIMIZER_BATCHINGCHECKER_H_

#include <ast_opt/ast/AbstractNode.h>
#include <queue>
#include <vector>

// TODO: add comments

class BatchingChecker {
  static AbstractNode *getLargestBatchableSubtree(AbstractExpr *expr);

  static bool isTransparentNode(AbstractNode *node);

  static bool isBatchingCompatible(AbstractNode *baseNode, AbstractNode *curNode);

  static bool isBatchableSubtree(AbstractNode *subtreeRoot);

  static std::vector<AbstractNode *> getChildren(AbstractNode *node);
};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_OPTIMIZER_BATCHINGCHECKER_H_
