#ifndef AST_OPTIMIZER_INCLUDE_UTILITIES_MULTIPLICATIVEDEPTHCALCULATOR_H_
#define AST_OPTIMIZER_INCLUDE_UTILITIES_MULTIPLICATIVEDEPTHCALCULATOR_H_

#include <unordered_map>
#include <string>
#include "Ast.h"

class MultiplicativeDepthCalculator {
 private:
  // A map of the computed multiplicative depths.
  std::unordered_map<std::string, int> multiplicativeDepths{};
  // A map of the computed reverse multiplicative depths.
  std::unordered_map<std::string, int> multiplicativeDepthsReversed{};
  // The maximum multiplicative depth, determined using the computed values in multiplicativeDepths.
  int maximumMultiplicativeDepth{};
  // A map of the initial multiplicative depths.
  std::unordered_map<std::string, int> initialMultiplicativeDepths{};

 public:
  explicit MultiplicativeDepthCalculator(Ast &ast);

  MultiplicativeDepthCalculator(Ast &ast, std::unordered_map<std::string, int> initialDepths);

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

  void precomputeMultDepths(Ast &ast);

  int getMaximumMultiplicativeDepth();

  int getInitialDepthOrNull(const std::string &uniqueNodeId);
};

#endif //AST_OPTIMIZER_INCLUDE_UTILITIES_MULTIPLICATIVEDEPTHCALCULATOR_H_
