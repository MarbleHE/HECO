#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_OPTIMIZER_BATCHINGCHECKER_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_OPTIMIZER_BATCHINGCHECKER_H_

#include <ast_opt/ast/AbstractNode.h>
#include <queue>
#include <vector>

class BatchingChecker {
 public:
  /// Takes an AbstractExpr expr and checks all subexpressions for the largest subtree that can be batched by
  /// descending the tree.
  /// \param expr The starting node to search for the largest batchable subtree.
  /// \return The root node of the largest batchable subtree.
  static AbstractNode *getLargestBatchableSubtree(AbstractExpr *expr);

  /// Determines if a subtree that has been detected to be batchable (see getLargestBatchableSubtree) is actually
  /// worthwhile to be batched because batchng also involves an overhead and thus does not make sense if cheap
  /// operations (e.g., addition) or only very few operands are involved.
  /// \param largestBatchableSubtreeRoot The root node of a subtree that has been detected to be batchable.
  /// \return True if the subtree rooted in node largestBatchableSubtreeRoot is worthwhile for being batched
  /// otherwise False.
  static bool shouldBeBatched(AbstractNode *largestBatchableSubtreeRoot);

 private:
  /// Determines whether the given node is transparent. We consider OperatorExpr nodes are transparent as we
  /// can expand expressions to "ignore" transparent nodes, e.g, 4 and 12*2 can be expanded to 4*1 and 12*2 that can
  /// be batched as [4 12] * [1 2] hence we consider the OperatorExpr (multiplication) as transparent.
  /// \param node The node that should be checked for transparency.
  /// \return True if this node is transparent otherwise False.
  static bool isTransparentNode(AbstractNode *node);

  /// Checks whether two given nodes are batching compatible.
  /// \param baseNode The node to compare with.
  /// \param curNode The node to compare with.
  /// \return True if baseNode and curNode are batching compatible.
  static bool isBatchingCompatible(AbstractNode *baseNode, AbstractNode *curNode);

  /// Determiens whether the node rooted in subtreeRoot is batchable.
  /// \param subtreeRoot The root node of the subtree that should be checked for batchability.
  /// \return True if the subtree rooted in subtreeRoot is batchable otherwise False.
  static bool isBatchableSubtree(AbstractNode *subtreeRoot);

  /// Returns the children of the given node that are relevant for verifying batching compatibility.
  /// \param node The node of that the children should be retrieved.
  /// \return The children of node that are required for checking batching compatibility.
  static std::vector<AbstractNode *> getChildren(AbstractNode *node);
};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_OPTIMIZER_BATCHINGCHECKER_H_
