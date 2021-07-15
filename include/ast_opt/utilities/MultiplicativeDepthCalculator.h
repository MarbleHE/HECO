#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_UTILITIES_MULTIPLICATIVEDEPTHCALCULATOR_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_UTILITIES_MULTIPLICATIVEDEPTHCALCULATOR_H_

#include <unordered_map>
#include <string>
#include "ast_opt/ast/AbstractNode.h"

struct DepthMapEntry {
  int multiplicativeDepth;
  int reverseMultiplicativeDepth;
  DepthMapEntry(int multiplicativeDepth, int reverseMultiplicativeDepth);
};

class MultiplicativeDepthCalculator {
 private:
  MultiplicativeDepthCalculator(AbstractNode &ast, std::unordered_map<std::string,
                                                                      DepthMapEntry> initialDepths);
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

 public:
  explicit MultiplicativeDepthCalculator(AbstractNode &ast);

  /// Calculates the multiplicative depth based on the definition given in
  /// [Aubry, P. et al.: Faster Homomorphic Encryption Is Not Enough: Improved Heuristic for Multiplicative Depth
  ///  Minimization of Boolean Circuits. (2019)].
  /// \return The multiplicative depth of the current node.
  int getMultDepthL(AbstractNode *n);

  /// Determine the value of this node for computing the multiplicative depth and reverse multiplicative depth,
  /// getMultDepthL() and getReverseMultDepthR(), respectively.
  /// \return Returns 1 iff this node is a LogicalExpr containing an AND operator, otherwise 0.
  static int depthValue(AbstractNode *n);

  /// Calculates the reverse multiplicative depth based on the definition given in
  /// [Aubry, P. et al.: Faster Homomorphic Encryption Is Not Enough: Improved Heuristic for Multiplicative Depth
  ///  Minimization of Boolean Circuits. (2019)].
  /// \return The reverse multiplicative depth of the current node.
  int getReverseMultDepthR(AbstractNode *n);

  void precomputeMultDepths(AbstractNode &ast);

  int getMaximumMultiplicativeDepth();

  DepthMapEntry getInitialDepthOrNull(AbstractNode *node);
};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_UTILITIES_MULTIPLICATIVEDEPTHCALCULATOR_H_
