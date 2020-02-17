#ifndef AST_OPTIMIZER_INCLUDE_OPTIMIZER_CONEREWRITER_H_
#define AST_OPTIMIZER_INCLUDE_OPTIMIZER_CONEREWRITER_H_

#include "AbstractNode.h"
#include <vector>
#include <random>
#include "Ast.h"
#include <utility>
#include "Operator.h"
#include "MultiplicativeDepthCalculator.h"

/// This class implements the Cone Rewriting method described in
/// [Aubry, P. et al.: Faster Homomorphic Encryption Is Not Enough: Improved Heuristic for Multiplicative Depth
/// Minimization of Boolean Circuits. (2019)].
class ConeRewriter {
private:
    Ast *ast;
    MultiplicativeDepthCalculator mdc;
    std::unordered_map<std::string, AbstractNode*> underlying_nodes;

    // ------------------------------------–
    // Internal (non-universal) helper methods
    // ------------------------------------–
    void rewriteCones(Ast &astToRewrite, const std::vector<AbstractNode *> &coneEndNodes);

    bool isCriticalNode(AbstractNode *n);

    int getMaxMultDepth(Ast &ast);

    AbstractNode *getBFSLastNonCriticalLeafNode();

    static void addElements(std::vector<AbstractNode *> &result, std::vector<AbstractNode *> newElements);

    static void flattenVectors(std::vector<AbstractNode *> &resultVector, std::vector<std::vector<AbstractNode *>> vectorOfVectors);

    static void reverseEdges(const std::vector<AbstractNode *> &nodes);

    std::vector<AbstractNode *> computeReducibleCones();

    static std::vector<AbstractNode *> sortTopologically(const std::vector<AbstractNode *> &nodes);

    std::vector<AbstractNode *> *getPredecessorOnCriticalPath(AbstractNode *v);

    int computeMinDepth(AbstractNode *v);

    // ------------------------------------–
    // Algorithms presented in the paper
    // ------------------------------------–
    /// Implements the recursive cone construction algorithm [see Algorithm 1, page 8]
    /// \param v The starting node for the cone construction procedure.
    /// \param minDepth The minimal multiplicative depth to which cone search will be performed.
    /// \return The cone to be rewritten, or an empty set if the cone ending at v cannot be reduced.
    std::vector<AbstractNode *> getReducibleCones(AbstractNode *v, int minDepth);

    std::vector<AbstractNode *> getReducibleCones();

    /// Calls and prints the output of getReducibleCones for each node in the given Ast.
    void getReducibleConesForEveryPossibleStartingNode(Ast &inputAst);

    /// Implements the multiplicative depth minimization heuristic [see Algorithm 2, page 10].
    Ast &applyMinMultDepthHeuristic();

    /// Implements the algorithm that constructs the graph C_{AND} of critical AND nodes [see paragraph 3.2, page 10].
    std::vector<AbstractNode *> getAndCriticalCircuit(std::vector<AbstractNode *> delta);

    /// Implements the cone selection algorithm [see Algorithm 3, page 11]
    static std::vector<AbstractNode *> selectCones(std::vector<AbstractNode *> cAndCkt);

    std::pair<AbstractNode *, AbstractNode *> getCriticalAndNonCriticalInput(LogicalExpr *n);

public:
    explicit ConeRewriter(Ast *ast);

    ConeRewriter(Ast *ast, MultiplicativeDepthCalculator &mdc);

    virtual ~ConeRewriter();

    /// This is the only entry point that should be used to apply cone rewriting.
    Ast &applyConeRewriting();
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
