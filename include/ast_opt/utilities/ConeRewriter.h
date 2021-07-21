#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_UTILITIES_CONEREWRITER_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_UTILITIES_CONEREWRITER_H_

#include "ast_opt/ast/BinaryExpression.h"
#include "ast_opt/ast/AbstractNode.h"


struct DepthMapEntry {
  int multiplicativeDepth;
  int reverseMultiplicativeDepth;
  DepthMapEntry(int multiplicativeDepth, int reverseMultiplicativeDepth);
};


class ConeRewriter {
 private:
  std::unique_ptr<AbstractNode> ast;
  // A map of the computed multiplicative depths.
  std::unordered_map<std::string, int> multiplicativeDepths{};
  // A map of the computed reverse multiplicative depths.
  std::unordered_map<std::string, int> multiplicativeDepthsReversed{};
  // The maximum multiplicative depth, determined using the computed values in multiplicativeDepths.
  int maximumMultiplicativeDepth{};
  // A map of the initial multiplicative depth:
  // - std::string: The variable's identifier for which this initial depth is associated to.
  // - DepthMapEntry: A struct containing the multiplicative and reverseMultiplicativeDepth.
  std::unordered_map<std::string, DepthMapEntry> initialMultiplicativeDepths{};

//  std::unordered_map<std::string, AbstractNode *> underlying_nodes;

  std::vector<AbstractNode *> getReducibleCones();

  std::unique_ptr<AbstractNode> rewriteCones(std::vector<AbstractNode *> &coneEndNodes);

//  std::pair<AbstractNode *, AbstractNode *> getCriticalAndNonCriticalInput(BinaryExpression *logicalExpr);
//  bool isCriticalNode(AbstractNode *n);
//
//  std::vector<AbstractNode *> getReducibleCones(AbstractNode *v, int minDepth);
//
//  std::vector<AbstractNode *> computeReducibleCones();
//
//  std::vector<AbstractNode *> *getPredecessorOnCriticalPath(AbstractNode *v);

 public:
  /// Create a cone rewriting object
  /// Can be used to rewrite multiple ASTs
  ConeRewriter() = default;

  /// Takes ownership of an AST, rewrites it and returns (potentially a different) rewritten AST
  /// \param ast AST to be rewritten (function takes ownership)
  /// \return a new AST that has been rewritten
  std::unique_ptr<AbstractNode> rewriteAst(std::unique_ptr<AbstractNode>&& ast);

  int computeMinDepth(AbstractNode *v);

  bool isCriticalNode(AbstractNode *n);

  /// Calculates the multiplicative depth based on the definition given in
  /// [Aubry, P. et al.: Faster Homomorphic Encryption Is Not Enough: Improved Heuristic for Multiplicative Depth
  ///  Minimization of Boolean Circuits. (2019)].
  /// \return The multiplicative depth of the current node.
  int getMultDepthL(AbstractNode *n);

  /// Calculates the reverse multiplicative depth based on the definition given in
  /// [Aubry, P. et al.: Faster Homomorphic Encryption Is Not Enough: Improved Heuristic for Multiplicative Depth
  ///  Minimization of Boolean Circuits. (2019)].
  /// \return The reverse multiplicative depth of the current node.
  int getReverseMultDepthR(AbstractNode *n);

  /// Determine the value of this node for computing the multiplicative depth and reverse multiplicative depth,
  /// getMultDepthL() and getReverseMultDepthR(), respectively.
  /// \return Returns 1 iff this node is a LogicalExpr containing an AND operator, otherwise 0.
  static int depthValue(AbstractNode *n);

  void precomputeMultDepths(AbstractNode &ast);

  int getMaximumMultiplicativeDepth();

  DepthMapEntry getInitialDepthOrNull(AbstractNode *node);
};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_UTILITIES_CONEREWRITER_H_
