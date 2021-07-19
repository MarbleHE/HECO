#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_UTILITIES_CONEREWRITER_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_UTILITIES_CONEREWRITER_H_

#include "ast_opt/ast/BinaryExpression.h"
#include "ast_opt/ast/AbstractNode.h"
#include "ast_opt/utilities/MultiplicativeDepthCalculator.h"

class ConeRewriter {
  private:
  ConeRewriter(AbstractNode *ast, MultiplicativeDepthCalculator mdc);
  AbstractNode *ast;
  std::unordered_map<std::string, AbstractNode *> underlying_nodes;
  MultiplicativeDepthCalculator mdc;


  void rewriteCones(AbstractNode &astToRewrite, const std::vector<AbstractNode *> &coneEndNodes);

  std::pair<AbstractNode *, AbstractNode *> getCriticalAndNonCriticalInput(BinaryExpression *logicalExpr);
  bool isCriticalNode(AbstractNode *n);



  std::vector<AbstractNode *> getReducibleCones(AbstractNode *v, int minDepth);

  std::vector<AbstractNode *> getReducibleCones();

  std::vector<AbstractNode *> computeReducibleCones();

  std::vector<AbstractNode *> *getPredecessorOnCriticalPath(AbstractNode *v);
};


#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_UTILITIES_CONEREWRITER_H_
