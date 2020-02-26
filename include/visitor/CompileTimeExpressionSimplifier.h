#ifndef AST_OPTIMIZER_INCLUDE_VISITOR_COMPILETIMEEXPRESSIONSIMPLIFIER_H_
#define AST_OPTIMIZER_INCLUDE_VISITOR_COMPILETIMEEXPRESSIONSIMPLIFIER_H_

#include "Visitor.h"
#include "EvaluationVisitor.h"
#include "BinaryExpr.h"
#include "LiteralFloat.h"
#include <queue>

/// Saves the information that we need to rewrite If statements like
///   if (condition) { a = 2; } else { a = 4; }
/// as simple expressions like
///   a = [condition]*2 + [1-condition]*4;
/// where condition is a LogicalExpr.
struct IfStatementResolver {
  AbstractExpr *factorIsTrue;
  AbstractExpr *factorIsFalse;

  explicit IfStatementResolver(AbstractExpr *ifStatementCondition) {
    // factorIsTrue = ifStatementCondition
    factorIsTrue = ifStatementCondition->clone(false)->castTo<AbstractExpr>();
    // factorIsFalse = [1-ifStatementCondition]
    factorIsFalse =
        new BinaryExpr(new LiteralInt(1),
                       OpSymb::subtraction,
                       ifStatementCondition->clone(false)->castTo<AbstractExpr>());
  };

  AbstractExpr *generateIfDependentValue(AbstractExpr *trueValue, AbstractExpr *falseValue) {
    // Build an expression like
    //   condition*trueValue + (1-b)*falseValue.
    // We need to handle the case where trueValue or/and falseValue are null because in that case the dependent
    // statement can be simplified by removing one/both operands of the binary expression.
    auto trueValueIsNull =
        dynamic_cast<AbstractLiteral *>(trueValue)!=nullptr && trueValue->castTo<AbstractLiteral>()->isNull();
    auto falseValueIsNull =
        dynamic_cast<AbstractLiteral *>(falseValue) && falseValue->castTo<AbstractLiteral>()->isNull();
    if (trueValueIsNull && falseValueIsNull) {
      // case: trueValue == 0 && falseValue == 0
      // return a cloned copy of trueValue because we cannot directly create a new object (e.g., LiteralInt) as we do
      // not exactly know which subtype of AbstractLiteral trueValue has
      // return "0" (where 0 is of the respective input type)
      return trueValue->clone(false)->castTo<AbstractExpr>();
    } else if (trueValueIsNull) {
      // case: trueValue == 0 && falseValue != 0 => value is 0 if the condition is True
      // return (1-b)*falseValue
      return new BinaryExpr(factorIsFalse, OpSymb::multiplication, falseValue);
    } else if (falseValueIsNull) {
      // case: trueValue != 0 && falseValue == 0 => value is 0 if the condition is False
      // return condition * trueValue
      return new BinaryExpr(factorIsTrue, OpSymb::multiplication, trueValue);
    }

    // default case: trueValue != 0 && falseValue != 0 => value is non-zero independent of what condition evaluates to
    // return condition*trueValue + (1-b)*falseValue.
    return new BinaryExpr(
        new BinaryExpr(factorIsTrue,
                       OpSymb::multiplication,
                       trueValue->clone(false)->castTo<AbstractExpr>()),
        OpSymb::addition,
        new BinaryExpr(factorIsFalse,
                       OpSymb::multiplication,
                       falseValue->clone(false)->castTo<AbstractExpr>()));
  }
};

class CompileTimeExpressionSimplifier : public Visitor {
 private:
  /// A EvaluationVisitor instance that is used to evaluate parts of the AST in order to simplify them.
  EvaluationVisitor evalVisitor;

  std::stack<IfStatementResolver *> ifResolverData;

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

  void storeEvaluatedNode(AbstractNode *node, const std::vector<AbstractExpr *> &evaluationResult);

  void storeEvaluatedNode(AbstractNode *node, AbstractExpr *evaluationResult);

  AbstractExpr *getFirstValue(AbstractNode *node);

  std::vector<AbstractLiteral *> evaluateNodeRecursive(AbstractNode *n,
                                                       std::unordered_map<std::string,
                                                                          AbstractLiteral *> valuesOfVariables);

  void handleBinaryExpressions(AbstractNode &binaryExpr, AbstractExpr *leftOperand, AbstractExpr *rightOperand);

  void visit(ParameterList &elem) override;

  std::unordered_map<std::string, AbstractLiteral *> getTransformedVariableMap();

  AbstractExpr *getFirstValueOrExpression(AbstractNode *node);

  static AbstractLiteral *getDefaultVariableInitializationValue(Types datatype);
};

#endif //AST_OPTIMIZER_INCLUDE_VISITOR_COMPILETIMEEXPRESSIONSIMPLIFIER_H_
