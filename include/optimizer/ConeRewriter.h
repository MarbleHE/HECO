#ifndef AST_OPTIMIZER_INCLUDE_OPTIMIZER_CONEREWRITER_H_
#define AST_OPTIMIZER_INCLUDE_OPTIMIZER_CONEREWRITER_H_

#include <Node.h>
#include <vector>
#include <random>
#include "Ast.h"

/// This class implements the Cone Rewriting method described in
/// [Aubry, P. et al.: Faster Homomorphic Encryption Is Not Enough: Improved Heuristic for Multiplicative Depth
/// Minimization of Boolean Circuits. (2019)].
class ConeRewriter {
private:
    // ------------------------------------–
    // Internal (non-universal) helper methods
    // ------------------------------------–
    static Ast &rewriteCones(std::vector<Node *> vector);

    static bool isCriticalNode(int lMax, Node *n);

    static int getMaxMultDepth(const Ast &ast);

    static Node *getNonCriticalLeafNode(Ast &ast);

    static void addElements(std::vector<Node *> &result, std::vector<Node *> newElements);

    static void flattenVectors(std::vector<Node *> &resultVector, std::vector<std::vector<Node *>> vectorOfVectors);

    static void reverseEdges(Ast &ast);

    static void reverseEdges(const std::vector<Node *> &nodes);

    static void validateCircuit(Ast &ast);

    static std::vector<Node *> computeReducibleCones(Ast &ast);

    static std::vector<Node *> sortTopologically(const std::vector<Node *> &nodes);

    static std::vector<Node *> *getCriticalPredecessors(Node *v);

    static int getMultDepthL(Node *v);

    static int getReverseMultDepthR(Node *v);

    static int depthValue(Node *n);

    static int computeMinDepth(Node *v);

    static std::vector<Node *> getAnc(Node *n);

    // ------------------------------------–
    // Algorithms presented in the paper
    // ------------------------------------–
    /// Implements the recursive cone construction algorithm [see Algorithm 1, page 8]
    /// \param v The starting node for the cone construction procedure.
    /// \param minDepth The minimal multiplicative depth to which cone search will be performed.
    /// \return The cone to be rewritten, or an empty set if the cone ending at v cannot be reduced.
    static std::vector<Node *> getReducibleCones(Node *v, int minDepth);

    static std::vector<Node *> getReducibleCones(Ast &ast);

    /// Calls and prints the output of getReducibleCones for each node in the given Ast
    static void getReducibleConesForEveryPossibleStartingNode(Ast &ast);

    /// Implements the multiplicative depth minimization heuristic [see Algorithm 2, page 10]

    static Ast &applyMinMultDepthHeuristic(Ast &ast);

    /// Implements the algorithm that constructs the graph C_{AND} of critical AND nodes [see paragraph 3.2, page 10].
    static std::vector<Node *> getAndCriticalCircuit(std::vector<Node *> delta);

    static std::vector<Node *> getAndCriticalCircuit(const Ast &ast);

    /// Implements the cone selection algorithm [see Algorithm 3, page 11]
    static std::vector<Node *> selectCones(std::vector<Node *> cAndCkt);

public:
    /// This is the only entry point that should be used to apply cone rewriting.
    static Ast &applyConeRewriting(Ast &ast);

    /// Returns the depth (L) and reverse depth (R), computed using the internal helper methods.
    friend std::pair<int, int> getDepthsLR(Node &n);
};

/// Returns a random element from a
/// Credits to Christopher Smith from stackoverflow, see https://stackoverflow.com/a/16421677/3017719.
template<typename Iter>
Iter select_randomly(Iter start, Iter end) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, std::distance(start, end) - 1);
    std::advance(start, dis(gen));
    return start;
}

#endif //AST_OPTIMIZER_INCLUDE_OPTIMIZER_CONEREWRITER_H_
