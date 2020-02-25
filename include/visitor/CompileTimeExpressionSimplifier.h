#ifndef AST_OPTIMIZER_INCLUDE_VISITOR_COMPILETIMEEXPRESSIONSIMPLIFIER_H_
#define AST_OPTIMIZER_INCLUDE_VISITOR_COMPILETIMEEXPRESSIONSIMPLIFIER_H_

#include "Visitor.h"
#include "EvaluationVisitor.h"
#include "BinaryExpr.h"
#include <queue>

struct IfStatementResolverData {
  AbstractExpr *factorIsTrue;
  AbstractExpr *factorIsFalse;

  explicit IfStatementResolverData(AbstractExpr *condition) {
    factorIsTrue = condition->clone(false)->castTo<AbstractExpr>();;
    factorIsFalse =
        new BinaryExpr(new LiteralInt(1),
                       OpSymb::subtraction,
                       condition->clone(false)->castTo<AbstractExpr>());
  };

  AbstractExpr *generateIfDependentValue(AbstractExpr *trueValue, AbstractExpr *falseValue) {
    // condition*trueValue + (1-b)*falseValue
    return new BinaryExpr(
        new BinaryExpr(factorIsTrue, OpSymb::multiplication, trueValue),
        OpSymb::addition,
        new BinaryExpr(factorIsFalse, OpSymb::multiplication, falseValue));
  }
};

class CompileTimeExpressionSimplifier : public Visitor {
 private:
  /// A EvaluationVisitor instance that is used to evaluate parts of the AST in order to simplify them.
  EvaluationVisitor evalVisitor;

  bool resolveIfStatementsActive = false;
  std::stack<IfStatementResolverData *> ifResolverData;

 public:
  CompileTimeExpressionSimplifier();

  /// Contains all nodes that could be evaluated during the simplification traversal with their associated evaluation
  /// result. This map serves
  /// - AbstractNode*: A reference to the evaluated node.
  /// - std::vector<AbstractLiteral*>: The node's evaluation result.
  std::unordered_map<AbstractNode *, std::vector<AbstractLiteral *>> evaluatedNodes;

  /// Stores the latest value of a variable while traversing through the AST.
  /// - std::string: The variable's identifier.
  /// - AbstractLiteral*: The variable's value.
  std::unordered_map<std::string, AbstractExpr *> variableValues;

  /// Contains pointer to those nodes for which full or partial evaluation could be performed and hence can be deleted
  /// at the end of this simplification traversal.
  /// For example, the expression BinaryExpr(LiteralInt(12), OpSymb::add, LiteralInt(42)) will be evaluated to 12+42=54.
  /// The node BinaryExpr (and all of its children) will be deleted and replaced by a new node LiteralInt(54).
  std::queue<AbstractNode *> nodesQueuedForDeletion;

  /** @defgroup visit Methods implementing the logic of the visitor for each node type.
  *  @{
  */

  void visit(AbstractNode &elem) override;
  void visit(AbstractExpr &elem) override;
  void visit(AbstractStatement &elem) override;
  void visit(BinaryExpr &elem) override;
  void visit(Block &elem) override;
  void visit(Call &elem) override;
  void visit(CallExternal &elem) override;
  void visit(Datatype &elem) override;
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
  void visit(UnaryExpr &elem) override;
  void visit(VarAssignm &elem) override;
  void visit(VarDecl &elem) override;
  void visit(Variable &elem) override;
  void visit(While &elem) override;
  void visit(Ast &elem) override;

  /** @} */ // End of visit group

  bool valueIsKnown(AbstractNode *abstractExpr);

  void storeEvaluatedNode(AbstractNode *node, const std::vector<AbstractLiteral *> &evaluationResult);

  void storeEvaluatedNode(AbstractNode *node, AbstractLiteral *evaluationResult);

  AbstractLiteral *getFirstValue(AbstractNode *node);

  std::vector<AbstractLiteral *> evaluateNodeRecursive(AbstractNode *n,
                                                       std::unordered_map<std::string,
                                                                          AbstractLiteral *> valuesOfVariables);

  void handleBinaryExpressions(AbstractNode &expr, AbstractExpr *leftOperand, AbstractExpr *rightOperand);
  void visit(ParameterList &elem) override;
  void moveChildIfNotEvaluable(AbstractNode *ifStatementsParent,
                               AbstractStatement *branchStatement);
  std::unordered_map<std::string, AbstractLiteral *> getTransformedVariableMap();
  AbstractExpr *getFirstValueOrExpression(AbstractNode *node);
};

#endif //AST_OPTIMIZER_INCLUDE_VISITOR_COMPILETIMEEXPRESSIONSIMPLIFIER_H_
