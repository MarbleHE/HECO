#ifndef AST_OPTIMIZER_INCLUDE_VISITOR_COMPILETIMEEXPRESSIONSIMPLIFIER_H_
#define AST_OPTIMIZER_INCLUDE_VISITOR_COMPILETIMEEXPRESSIONSIMPLIFIER_H_

#include "Visitor.h"
#include "EvaluationVisitor.h"
#include "ArithmeticExpr.h"
#include "LiteralFloat.h"

class CompileTimeExpressionSimplifier : public Visitor {
 private:
  /// A EvaluationVisitor instance that is used to evaluate parts of the AST in order to simplify them.
  EvaluationVisitor evalVisitor;

 public:
  CompileTimeExpressionSimplifier();

  /// Contains all nodes that could be evaluated during the simplification traversal with their associated evaluation
  /// result. This map is needed in addition to variableValues because we need to store the results of partially
  /// evaluated expressions, e.g., for int result = a+23*6 this map would contain the partial result 23*6=138 for the
  /// rhs operand that can than be used to simplify the assignment to: result = a+138.
  /// - AbstractNode*: A reference to the evaluated node.
  /// - std::vector<AbstractLiteral*>: The node's evaluation result.
  std::unordered_map<AbstractNode *, std::vector<AbstractExpr *>> evaluatedNodes;

  /// Stores the latest value of a variable while traversing through the AST.
  /// - std::string: The variable's identifier.
  /// - AbstractExpr*: The variable's value.
  std::unordered_map<std::string, AbstractExpr *> variableValues;

  /// Contains pointer to those nodes for which full or partial evaluation could be performed and hence can be deleted
  /// at the end of this simplification traversal.
  /// For example, the expression ArithmeticExpr(LiteralInt(12), OpSymb::add, LiteralInt(42)) will be evaluated to 12+42=54.
  /// The node ArithmeticExpr (and all of its children) will be deleted and replaced by a new node LiteralInt(54).
  std::queue<AbstractNode *> nodesQueuedForDeletion;

  /** @defgroup visit Methods implementing the logic of the visitor for each node type.
  *  @{
  */

  void visit(AbstractNode &elem) override;
  void visit(AbstractExpr &elem) override;
  void visit(AbstractStatement &elem) override;
  void visit(ArithmeticExpr &elem) override;
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

  void storeEvaluatedNode(AbstractNode *node, const std::vector<AbstractExpr *> &evaluationResult);

  void storeEvaluatedNode(AbstractNode *node, AbstractExpr *evaluationResult);

  AbstractExpr *getFirstValue(AbstractNode *node);

  std::vector<AbstractLiteral *> evaluateNodeRecursive(AbstractNode *n,
                                                       std::unordered_map<std::string,
                                                                          AbstractLiteral *> valuesOfVariables);

  void handleBinaryExpressions(AbstractNode &arithmeticExpr, AbstractExpr *leftOperand, AbstractExpr *rightOperand);

  void visit(ParameterList &elem) override;

  std::unordered_map<std::string, AbstractLiteral *> getTransformedVariableMap();

  AbstractExpr *getFirstValueOrExpression(AbstractNode *node);

  static AbstractLiteral *getDefaultVariableInitializationValue(Types datatype);

  /// A helper method to transform an If statement into a dependent assignment, for example:
  ///     if (condition) { x = trueValue; } else { x = falseValue; }
  /// is converted into
  ///     x = condition*trueValue + (1-b)*falseValue.
  /// This method takes the required parts of this expression, the condition, the value that should be assigned in case
  /// that the condition evaluates to True (trueValue) or to False (falseValue). It then generates and returns the
  /// following expression: condition*trueValue + (1-b)*falseValue.
  /// The method also considers the case where trueValue and/or falseValue are null and appropriately removes the
  /// irrelevant subtree from the resulting expression.
  /// \param condition The condition the assignment depends on, e.g., the condition of the If statement.
  /// \param trueValue The value to be used for the case that the condition evaluates to True.
  /// \param falseValueThe value to be used for the case that the condition evaluates to False.
  /// \return A arithmetic expression of the form condition*trueValue + (1-b)*falseValue.
  static AbstractExpr *generateIfDependentValue(AbstractExpr *condition,
                                                AbstractExpr *trueValue,
                                                AbstractExpr *falseValue);
  void simplifyNestedBinaryExpressions(AbstractExpr *nestedExprRoot);
};

#endif //AST_OPTIMIZER_INCLUDE_VISITOR_COMPILETIMEEXPRESSIONSIMPLIFIER_H_
