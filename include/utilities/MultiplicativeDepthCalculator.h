#ifndef AST_OPTIMIZER_INCLUDE_UTILITIES_MULTIPLICATIVEDEPTHCALCULATOR_H_
#define AST_OPTIMIZER_INCLUDE_UTILITIES_MULTIPLICATIVEDEPTHCALCULATOR_H_

#include <map>
#include <string>

class MultiplicativeDepthCalculator {
private:
    std::unordered_map<std::string, int> multiplicativeDepths{};
    std::unordered_map<std::string, int> multiplicativeDepthsReversed{};
    int maximumMultiplicativeDepth{};

public:
    explicit MultiplicativeDepthCalculator(Ast &ast);

    /// Calculates the multiplicative depth based on the definition given in
    /// [Aubry, P. et al.: Faster Homomorphic Encryption Is Not Enough: Improved Heuristic for Multiplicative Depth
    ///  Minimization of Boolean Circuits. (2019)].
    /// \return The multiplicative depth of the current node.
    int getMultDepthL(Node *n);

    /// Determine the value of this node for computing the multiplicative depth and reverse multiplicative depth,
    /// getMultDepthL() and getReverseMultDepthR(), respectively.
    /// \return Returns 1 iff this node is a LogicalExpr containing an AND operator, otherwise 0.
    static int depthValue(Node *n);

    /// Calculates the reverse multiplicative depth based on the definition given in
    /// [Aubry, P. et al.: Faster Homomorphic Encryption Is Not Enough: Improved Heuristic for Multiplicative Depth
    ///  Minimization of Boolean Circuits. (2019)].
    /// \return The reverse multiplicative depth of the current node.
    int getReverseMultDepthR(Node *n);

    void precomputeMultDepths(Ast &ast);

    int getMaximumMultiplicativeDepth();
};

#endif //AST_OPTIMIZER_INCLUDE_UTILITIES_MULTIPLICATIVEDEPTHCALCULATOR_H_
