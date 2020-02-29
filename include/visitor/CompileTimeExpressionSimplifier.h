#ifndef AST_OPTIMIZER_INCLUDE_VISITOR_COMPILETIMEEXPRESSIONSIMPLIFIER_H_
#define AST_OPTIMIZER_INCLUDE_VISITOR_COMPILETIMEEXPRESSIONSIMPLIFIER_H_

#include <NodeUtils.h>
#include "Visitor.h"
#include "EvaluationVisitor.h"
#include "ArithmeticExpr.h"
#include "LiteralFloat.h"
#include "Variable.h"

typedef std::variant<OpSymb::ArithmeticOp, OpSymb::LogCompOp, OpSymb::UnaryOp> OpSymbolVariant;

struct BinaryExpressionAcc {
  std::variant<OpSymb::ArithmeticOp, OpSymb::LogCompOp, OpSymb::UnaryOp> operatorSymbol;
  std::vector<AbstractExpr *> operands;
  AbstractExpr *lastVisitedSubtree;
  unsigned long numberOfReducedNodes = 0;

  void setLastVisitedSubtree(AbstractExpr *node) {
    lastVisitedSubtree = node;
  }

  void clear() {
    operands.clear();
    lastVisitedSubtree = nullptr;
    numberOfReducedNodes = 0;
  }

  bool subtreeIsSimplified() {
    return numberOfReducedNodes > 0;
  }

  static bool isSupportedOperator(std::variant<OpSymb::ArithmeticOp, OpSymb::LogCompOp, OpSymb::UnaryOp> opSymbol) {
    // all commutative operators
    static const std::vector<OpSymb::LogCompOp> arithmeticOps =
        {OpSymb::logicalAnd, OpSymb::logicalOr, OpSymb::logicalXor};
    static const std::vector<OpSymb::ArithmeticOp> logicalOps =
        {OpSymb::addition, OpSymb::multiplication};

    // accumulator approach only works for commutative operators
    if (std::holds_alternative<OpSymb::ArithmeticOp>(opSymbol)) {
      return std::find(arithmeticOps.begin(), arithmeticOps.end(), std::get<OpSymb::ArithmeticOp>(opSymbol))
          !=arithmeticOps.end();
    } else if (std::holds_alternative<OpSymb::LogCompOp>(opSymbol)) {
      return std::find(logicalOps.begin(), logicalOps.end(), std::get<OpSymb::LogCompOp>(opSymbol))
          !=logicalOps.end();
    }
    return false;
  }

  AbstractNode *getSimplifiedSubtree() {
    return createMultDepthBalancedTreeFromInputs(operands, operatorSymbol);
  }

  void evaluateLiterals() {
    auto isLiteral = [](AbstractExpr *expr) { return dynamic_cast<AbstractLiteral *>(expr)!=nullptr; };
    if (std::count_if(operands.begin(), operands.end(), isLiteral) > 1) {
      // update numberOfReducedNodes
      numberOfReducedNodes = numberOfReducedNodes + (operands.size() - 1);

      // Credits to T.C. from stackoverflow.com (https://stackoverflow.com/a/32155973/3017719)
      // partition: all elements that should not be moved come before
      std::vector<AbstractExpr *> extractedLiterals;
      auto p = std::stable_partition(operands.begin(), operands.end(),
                                     [&](const auto &x) { return !isLiteral(x); });
      // range insert with move
      extractedLiterals.insert(extractedLiterals.end(), std::make_move_iterator(p),
                               std::make_move_iterator(operands.end()));
      // erase the moved-from elements
      operands.erase(p, operands.end());

      // build tree consisting of literals only and evaluate it
      auto treeRoot = createMultDepthBalancedTreeFromInputs(extractedLiterals, operatorSymbol);
      EvaluationVisitor ev;
      treeRoot->accept(ev);

      // add the evaluation result as new literal
      operands.push_back(ev.getResults().front());
    }
  }

  void addOperands(std::vector<AbstractExpr *> operandsToBeAdded) {
    // we are traversing from leaf to root, hence don't add any binary expressions as they by itself already added
    // their operands to BinaryExpressionAcc
    auto it = std::remove_if(operandsToBeAdded.begin(), operandsToBeAdded.end(), [&](AbstractExpr *ae) {
      return dynamic_cast<AbstractBinaryExpr *>(ae)!=nullptr;
    });
    //
    operands.insert(operands.end(), operandsToBeAdded.begin(), it);
    evaluateLiterals();
  }

  void removeOperandsAndSetNewSymbol(std::variant<OpSymb::ArithmeticOp, OpSymb::LogCompOp, OpSymb::UnaryOp> newSymbol) {
    operands.clear();
    setOperator(newSymbol);
  }

  bool containsOperands() {
    return !operands.empty();
  }

  void setOperator(std::variant<OpSymb::ArithmeticOp, OpSymb::LogCompOp, OpSymb::UnaryOp> newOperatorSymbol) {
    operatorSymbol = newOperatorSymbol;
  }

  [[nodiscard]] const std::variant<OpSymb::ArithmeticOp,
                                   OpSymb::LogCompOp,
                                   OpSymb::UnaryOp> &getOperatorSymbol() const {
    return operatorSymbol;
  }

};

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
  std::deque<AbstractNode *> nodesQueuedForDeletion;

  BinaryExpressionAcc binaryExpressionAccumulator;

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

  void handleBinaryExpressions(AbstractBinaryExpr &arithmeticExpr,
                               AbstractExpr *leftOperand,
                               AbstractExpr *rightOperand);

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

  void cleanUpAfterStatementVisited(AbstractNode *stat, bool enqueueStatementForDeletion = false);
  void addVariableValue(const std::string &variableIdentifier, AbstractExpr *valueAnyLiteralOrAbstractExpr);
  bool isQueuedForDeletion(const AbstractNode *node);
};

#endif //AST_OPTIMIZER_INCLUDE_VISITOR_COMPILETIMEEXPRESSIONSIMPLIFIER_H_
