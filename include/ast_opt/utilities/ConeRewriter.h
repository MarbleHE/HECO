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
  /// Implements the recursive cone construction algorithm [see Algorithm 1, page 8]:
  /// Multiplicative depth-2 cone rewrite operators require that at least one input of then v_i-nodes (see Fig 2: paper) is
  /// non-critical. In the case when both inputs are critical we can explore the input nodes of v_1 and build a  multiplicative depth-3 cone (and so on).
  ///
  /// The procedure recursively explores the set of critical predecessor nodes starting from node and incrementally constructs a reducible cone.
  /// (reducible = can reduce multiplicative depth)
  /// If the mini- mal multiplicative depth to explore is reached or at least one predecessor of an AND node v is not critical
  /// then the exploration stops. Otherwise there are two possibilities as a function of node v type:
  ///
  /// 1. AND node: If at least one predecessor is reducible then the cone corresponding to this predecessor (or a random one if both are reducible)
  /// is added to the result, otherwise the exploration is complete.
  ///
  /// 2. XOR node If both predecessors are reducible then the respective cones are added to the result, otherwise exploration is also complete.
  ///
  /// The procedure is called on a circuit node v. If the procedure returns an empty set then the cone ending at v cannot be reduced.
  /// Otherwise the procedure output represents the cone to be rewritten and it ensures that the multiplicative depth of this cone can be reduced.
  /// We use a minDepth value equal to l(p) + 1, where p is the non-critical input of node v.
  ///
  /// the ConeRec procedure when applied to the ending node of a reducible multiplicative depth-2 cone will find exactly that cone.
  /// In the case when no reducible multiplicative depth-2 cone ending at v exists the ConeRec procedure will return a cone with a multiplicative depth larger than 2.
  /// Rewriting such a cone is very similar to the depth-2 cone rewriting method presented previ- ously. The multiplicative depth cone
  /// rewriting is a powerful tool for minimizing the multiplicative depth of Boolean circuits.
  ///
  /// \param root node defining the ast
  /// \param v Starting node for the cone construction procedure.
  /// \param minDepth The minimal multiplicative depth to which cone search will be performed.
  /// \return a reducible cone
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

  /// Creates the graph C^{AND} from Section 3.2 (p 10) from [Aubry, P. et al.: Faster Homomorphic Encryption Is Not Enough: Improved Heuristic for Multiplicative Depth
  ///  Minimization of Boolean Circuits. (2019)].
  /// Idea: We want to find a minimal size set \Delta_{min} of cones such that each critical path contains the ending node of at least one cone from this set
  /// (to guarantee that the overall multiplicative depth will in fact decrease)
  /// For this we need to constrcut a graph C^{AND} containing ALL the crtical end nodes. Two AND nodes are connected in C^{AND}
  /// if there is a depth-2 critical path between them in the initial circuit.
  /// \param root
  /// \param delta
  /// \return
  static std::vector<AbstractNode *> getAndCriticalCircuit(AbstractNode &root, std::vector<AbstractNode *> delta);

  ///  Implements the cone selection algorithm [see Algorithm 3, page 11].
  /// //TODO: understand
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

  /// Computes minDepth parameter required for cone selection algorithm
  /// We use a minDepth value equal to l(p) + 1, where p is the non-critical input of node v.
  /// \param v node
  /// \return minDepth value
  int computeMinDepth(AbstractNode *v);

  /// Returns the 'overall multiplicative depth' l^{max}, i.e the maximal multiplicative depth of its nodes.
  /// l^{max} = max_{v \in V} l(v) = max_{v \in V} r(v)
  /// \param root
  /// \return
  static int getOverallMultDepth(AbstractNode *root);

  /// Returns true if a node n is critical, i.e. if l(n) + r(n) = l^{max}, false otherwies
  /// \param n node of the AST
  /// \return bool
  bool isCriticalNode(AbstractNode *n);


  /// Calculates the multiplicative depths l(n) for a node n of an AST starting at root (output) based on the definition given in
  /// [Aubry, P. et al.: Faster Homomorphic Encryption Is Not Enough: Improved Heuristic for Multiplicative Depth
  ///  Minimization of Boolean Circuits. (2019)].
  /// The multiplicative depth is the max number of AND gates on any path beginning by an input node (leaf of AST) and ending
  /// in the node n.
  /// \param n Node of an AST
  /// \param map (Optional) map already containing precomputed values
  /// \return mult depth of the node n
  int computeMultDepthL(AbstractNode *n, std::unordered_map<std::string, int> map = {});


  /// Returns multiplicative depth as precomputed in computeMultDepthR for a given node
  /// \param multiplicativeDepths Mapping between nodes and their reversed depth
  /// \param n  Node to consider
  /// \return The multiplicative depth of the current node.
  int getMultDepth(std::unordered_map<std::string, int> multiplicativeDepths, AbstractNode &n);

  /// Compute the (reverse) multiplicative depths r(v) for all nodes v of an AST starting at root based on the definition given in
  /// [Aubry, P. et al.: Faster Homomorphic Encryption Is Not Enough: Improved Heuristic for Multiplicative Depth
  /// The reverse multiplicative depth is the maximum number of AND gates on any path beginning by a parent of n and ending with a
  /// root node (output).
  /// \param root
  /// \param map (Optional) map already containing precomputed values
  /// \return map containing all reverse mult depths of the AST
  std::unordered_map<std::string, int> computeReverseMultDepth(AbstractNode &root,
                                                                std::unordered_map<std::string, int> map = {});

  /// Returns multiplicative depth as precomputed in computeReverseMultDepth for a given node
  /// \param multiplicativeDepthsReversed Mapping between nodes and their reversed depth
  /// \param n  Node to consider
  /// \return The reverse multiplicative depth of the current node.
  int getReverseMultDepth(std::unordered_map<std::string, int> multiplicativeDepthsReversed, AbstractNode *n);

  /// Determine the value of this node (used for computing the multiplicative depth and reverse multiplicative depth)
  /// Returns 1 if n AND node, 0 otherwise.
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
  int getMultDepth(AbstractNode *n);
};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_UTILITIES_CONEREWRITER_H_
