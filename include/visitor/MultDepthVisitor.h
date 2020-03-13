#ifndef AST_OPTIMIZER_INCLUDE_VISITOR_MULTDEPTHVISITOR_H_
#define AST_OPTIMIZER_INCLUDE_VISITOR_MULTDEPTHVISITOR_H_

#include <map>
#include <string>
#include "AbstractNode.h"
#include "Visitor.h"
#include <utility>

class MultDepthVisitor : public Visitor {
 private:
  /// depthsPerVariable stores the depth for each variable (std::string is the associated variable identifier) during
  /// the ASTs traversal. At the end of the traversal, the structure contains the maximum depth for each variable.
  std::unordered_map<std::string, int> depthsPerVariable;
  /// depthsPerStatement stores the depth for each statement with its corresponding assignment target (std::string)
  /// and its current depth (int). It allows to determine the depth for each statement in the AST.
  std::map<AbstractStatement *, std::pair<std::string, int>> depthsPerStatement;
  bool verbose = false;

  int getDepth(const std::string &nodeName);

 public:
  explicit MultDepthVisitor(bool b = false);

  int getMaxDepth();

  void visit(ArithmeticExpr &elem) override;

  void visit(Block &elem) override;

  void visit(Call &elem) override;

  void visit(CallExternal &elem) override;

  void visit(For &elem) override;

  void visit(Function &elem) override;

  void visit(FunctionParameter &elem) override;

  void visit(If &elem) override;

  void visit(LiteralBool &elem) override;

  void visit(LiteralInt &elem) override;

  void visit(LiteralString &elem) override;

  void visit(LiteralFloat &elem) override;

  void visit(LogicalExpr &elem) override;

  void visit(Operator &elem) override;

  void visit(Return &elem) override;

  void visit(Rotate &elem) override;

  void visit(UnaryExpr &elem) override;

  void visit(VarAssignm &elem) override;

  void visit(VarDecl &elem) override;

  void visit(Variable &elem) override;

  void visit(While &elem) override;

  void visit(Ast &elem) override;

  void
  analyzeMultiplicativeDepth(const std::string &varIdentifier, AbstractStatement *stmt, AbstractExpr *initializer);

  void updateDepthStructures(AbstractStatement *stmt, const std::string &varIdentifier, int depth);

  void visit(Datatype &elem) override;

  void visit(ParameterList &elem) override;

  void visit(Transpose &elem) override;

  void visit(GetMatrixElement &elem) override;
};

#endif //AST_OPTIMIZER_INCLUDE_VISITOR_MULTDEPTHVISITOR_H_
