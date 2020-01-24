#ifndef AST_OPTIMIZER_INCLUDE_OPTIMIZER_CONEREWRITER_H_
#define AST_OPTIMIZER_INCLUDE_OPTIMIZER_CONEREWRITER_H_

#include <Node.h>
#include <vector>
#include <random>
#include <Ast.h>

class ConeRewriter {
  /// Implements the recursive cone construction algorithm.
  /// See Algorithm 1 in [Aubry, P. et al.: Faster Homomorphic Encryption Is Not Enough:
  /// Improved Heuristic for Multiplicative Depth Minimization of Boolean Circuits. (2019)].
  std::vector<Node*> getReducibleCones(Node* v, int minDepth);
  std::vector<Node*> getReducibleCones(Node* v);

  static void addElements(std::vector<Node*> &result, std::vector<Node*> newElements);

  static void flattenVectors(std::vector<Node*> &resultVector, std::vector<std::vector<Node*>> vectorOfVectors);

  static std::vector<Node*>* getCriticalPredecessors(Node* v);

  // Credits to Christopher Smith from stackoverflow, see https://stackoverflow.com/a/16421677/3017719.
  template<typename Iter, typename RandomGenerator>
  Iter select_randomly(Iter start, Iter end, RandomGenerator &g) {
    std::uniform_int_distribution<> dis(0, std::distance(start, end) - 1);
    std::advance(start, dis(g));
    return start;
  }

  // Credits to Christopher Smith from stackoverflow, see https://stackoverflow.com/a/16421677/3017719.
  template<typename Iter>
  Iter select_randomly(Iter start, Iter end) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    return select_randomly(start, end, gen);
  }

 public:
  static bool isValidCircuit(Ast &ast);
  static int getMultDepth(Node* v);
  static void reverseEdges(Ast &ast);
  static std::vector<Node*>* getPred(Node* n);
};

#endif //AST_OPTIMIZER_INCLUDE_OPTIMIZER_CONEREWRITER_H_
