#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_UTILITIES_CONEREWRITER_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_UTILITIES_CONEREWRITER_H_

#include "ast_opt/ast/BinaryExpression.h"
#include "ast_opt/ast/AbstractNode.h"
#include "ast_opt/utilities/MultiplicativeDepthCalculator.h"

class ConeRewriter {
 private:
  std::unique_ptr<AbstractNode> ast;
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


};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_UTILITIES_CONEREWRITER_H_
