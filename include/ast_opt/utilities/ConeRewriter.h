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

  // A map of the computed multiplicative depths.
  //  std::unordered_map<std::string, int> multiplicativeDepths{};
  //  // A map of the computed reverse multiplicative depths.
  //  std::unordered_map<std::string, int> multiplicativeDepthsReversed{};
  //  // The maximum multiplicative depth, determined using the computed values in multiplicativeDepths.
  //  int maximumMultiplicativeDepth{};
  //  // A map of the initial multiplicative depth:
  //  // - std::string: The variable's identifier for which this initial depth is associated to.
  //  // - DepthMapEntry: A struct containing the multiplicative and reverseMultiplicativeDepth.
  //  std::unordered_map<std::string, DepthMapEntry> initialMultiplicativeDepths{};
  //  std::unordered_map<std::string, AbstractNode *> underlying_nodes;


  /// Identify reducible cones in the (sub)tree defined by root
  /// Internal function, used recursively
  /// If the multdepth of v is  mindepth, then it returns {}
  /// If the multdepth v is greater or smaller than mindepth //TODO: CAN IT EVEN BE SMALLER THAN MINDEPTH?
  /// then it searches for "predecessor" (i.e. children in our view) nodes p
  /// for which l(p) == l(v) - d(v) where l(x) is the multdepth of x and d(x) = 1 iff it's an AND node.
  /// TODO: Move to multdepth function: l(x) is 0 if it doesn't have children, otherwise it's max_{child c} l(c) + d(x)
  /// If the size of the set of all such p is smaller than 2 and d(v) == 1 (i.e., it's an AND)
  /// then we return {v}
  /// Otherwise: We recursively call this function for each p from the set (using the same mindepth),
  /// We take the union of all result sets. If the result set is not empty
  /// and d(v) == 0 then we randomly select a node delta from the set and return {delta, v}
  /// otherwise, if the set is not empty and d(v) == 1, then we check if the size of the set is equal to
  /// the size of the set of all nodes p on which we did the recursive calls.
  ///     If this is true, we return TODO: WHAT?
  /// otherwise, if the result set is empty, we return {}
  /// \param root Root node of the AST
  /// \param v
  /// \param minDepth
  /// \return
  static std::vector<AbstractNode *> getReducibleCones(AbstractNode &root, AbstractNode *v, int minDepth);

  /// TODO: IMPLEMENT & DOCUMENT
  /// \param v
  /// \return
  //std::vector<AbstractNode *> *getPredecessorOnCriticalPath(AbstractNode& root, AbstractNode *v);

 public:
  /// Create a cone rewriting object
  /// Can be used to rewrite multiple ASTs
  ConeRewriter() = default;

  /// Takes ownership of an AST, rewrites it and returns (potentially a different) rewritten AST
  /// \param ast AST to be rewritten (function takes ownership)
  /// \return a new AST that has been rewritten
  static std::unique_ptr<AbstractNode> rewriteAst(std::unique_ptr<AbstractNode> &&ast);

  /// Identify reducible cones in the (sub)tree defined by root
  /// \param root Root node of the AST
  /// \return A vector of nodes, each node defines a cone starting at root and ending at the node
  static std::vector<AbstractNode *> getReducibleCones(AbstractNode &root);

  /// TODO: Document
  /// \param root
  /// \param delta
  /// \return
  static std::vector<AbstractNode *> getAndCriticalCircuit(AbstractNode &root, std::vector<AbstractNode *> delta);

  /// TODO: Document
  /// \param root
  /// \param cAndCkt
  /// \return
  static std::vector<AbstractNode *> selectCones(AbstractNode &root, std::vector<AbstractNode *> cAndCkt);

  /// TODO: Document
  /// \param ast
  /// \param coneEndNodes
  /// \return
  static std::unique_ptr<AbstractNode> rewriteCones(std::unique_ptr<AbstractNode> &&ast,
                                                    std::vector<AbstractNode *> &coneEndNodes);

  /////////////////////////////////////////////////////
  /// UTILITY FUNCTIONS FOR MULTDEPTH CALCULATION /////
  /////////////////////////////////////////////////////

  /// TODO: Document
  /// \param v
  /// \return
  int computeMinDepth(AbstractNode *v);

  /// TODO: Document -when is a node critical?
  /// \param n
  /// \return
  bool isCriticalNode(AbstractNode *n);

  /// Calculates the multiplicative depth based on the definition given in
  /// [Aubry, P. et al.: Faster Homomorphic Encryption Is Not Enough: Improved Heuristic for Multiplicative Depth
  ///  Minimization of Boolean Circuits. (2019)].
  /// \param n  Node to consider
  /// \return The multiplicative depth of the current node.
  int getMultDepthL(AbstractNode *n);

  /// Compute the (reverse) multiplicative depths for an AST starting at root
  /// \param root
  /// \param map (Optional) map already containing precomputed values
  /// \return
  std::unordered_map<std::string, int> computeReverseMultDepthR(AbstractNode &root,
                                                                std::unordered_map<std::string, int> map = {});

  /// Calculates the reverse multiplicative depth based on the definition given in
  /// [Aubry, P. et al.: Faster Homomorphic Encryption Is Not Enough: Improved Heuristic for Multiplicative Depth
  ///  Minimization of Boolean Circuits. (2019)].
  /// \param multiplicativeDepthsReversed Mapping between nodes and their reversed depth
  /// \param n  Node to consider
  /// \return The reverse multiplicative depth of the current node.
  int getReverseMultDepthR(std::unordered_map<std::string, int> multiplicativeDepthsReversed, AbstractNode *n);

  /// Determine the value of this node for computing the multiplicative depth and reverse multiplicative depth,
  /// getMultDepthL() and getReverseMultDepthR(), respectively.
  /// \param n  Node to consider
  /// \return Returns 1 iff this node is a LogicalExpr containing an AND operator, otherwise 0.
  static int depthValue(AbstractNode *n);

//  void precomputeMultDepths(AbstractNode *ast);
//  int getMaximumMultiplicativeDepth();//
//  DepthMapEntry getInitialDepthOrNull(AbstractNode *node);
//  std::vector<AbstractNode *> *getPredecessorOnCriticalPath(AbstractNode *v); // here getSuccessor
//  std::pair<AbstractNode *, AbstractNode *> getCriticalAndNonCriticalInput(BinaryExpression *logicalExpr);
//  std::pair<AbstractNode *, AbstractNode *> getCriticalAndNonCriticalInput(LogicalOp *logicalExpr);
//  std::pair<AbstractNode *, AbstractNode *> getCriticalAndNonCriticalInput(BinaryExpression *logicalExpr);
//  bool isCriticalNode(AbstractNode *n);
};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_UTILITIES_CONEREWRITER_H_
