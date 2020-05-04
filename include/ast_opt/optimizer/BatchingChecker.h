#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_OPTIMIZER_BATCHINGCHECKER_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_OPTIMIZER_BATCHINGCHECKER_H_

#include <ast_opt/ast/AbstractNode.h>
#include <queue>
#include <vector>

// TODO: add comments

class BatchingChecker {

  static bool isTransparentNode(AbstractNode *node);

  static bool isBatchingCompatible(AbstractNode *baseNode, AbstractNode *curNode);

  static bool isBatchableSubtree(AbstractNode *subtreeRoot);

  static std::vector<AbstractNode *> getChildren(AbstractNode *node);

 public:
  static AbstractNode *getLargestBatchableSubtree(AbstractExpr *expr);
};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_OPTIMIZER_BATCHINGCHECKER_H_
