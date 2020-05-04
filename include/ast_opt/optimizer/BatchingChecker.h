#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_OPTIMIZER_BATCHINGCHECKER_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_OPTIMIZER_BATCHINGCHECKER_H_

#include <ast_opt/ast/AbstractNode.h>
#include <queue>
class BatchingChecker {

 public:
  static void determineBatchability(AbstractNode *startNode);
};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_OPTIMIZER_BATCHINGCHECKER_H_
