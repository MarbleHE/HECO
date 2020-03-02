#ifndef AST_OPTIMIZER_INCLUDE_VISITOR_COMPILETIMEEXPRESSIONSIMPLIFIER_H_
#define AST_OPTIMIZER_INCLUDE_VISITOR_COMPILETIMEEXPRESSIONSIMPLIFIER_H_

#include <unordered_set>
#include "NodeUtils.h"
#include "Visitor.h"
#include "EvaluationVisitor.h"
#include "ArithmeticExpr.h"
#include "LiteralFloat.h"
#include "Variable.h"

/// This struct handles the accumulation of binary expression operands in order to simplify them.
/// For example the nested arithmetic expression
///     a + (23 - (11 + (21 + b)))
/// can be simplified to
///     a + (33 + b).
/// This accumulation considers commutative operators (e.g., addition, multiplication, logical-AND/-OR-XOR) only.
/// For that simplification, all operands, i.e., variables and literals of binary expressions with the same operator are
/// accumulated in the operands vector by adding them using addOperands(...). The operator is applied automatically on
/// the literals after adding a new operand such that there is always at the most one literal in the operands vector.
/// After encountering the first binary expression with another operator, it must be checked whether the
/// BinaryExpressionAcc could simplify the subtree using subtreeIsSimplified(). If yes, the simplified subtree can be
/// generated using getSimplifiedSubtree().
struct BinaryExpressionAcc {

  /// The operator of the accumulated operands.
  OpSymbolVariant operatorSymbol;

  /// The operands collected in this subtree (nested binary expressions). This represents the operands of the simplified
  /// tree as after adding any operand, the literals collected so far are combined using the operator specified by the
  /// current operatorSymbol.
  std::vector<AbstractExpr *> operands;

  /// The last visited binary expression. This information is needed to know where to attach to the newly created
  /// subtree of simplified binary expressions.
  AbstractExpr *lastVisitedBinaryExp;

  /// The first visited binary expression for which operands were collected. This information is needed when generating
  /// the simplified subtree consisting of the collected operands. If the firstVisitedBinaryExp had any binary
  /// expressions, then these will be attached to the simplified subtree.
  AbstractExpr *firstVisitedBinaryExp;

  /// The number of operands that could be reduced. This information is used to determine whether the
  /// BinaryExpressionAcc could simplify the binary expressions at all.
  unsigned long numberOfReducedNodes = 0;

  /// Sets a new value to lastVisitedBinaryExp that indicates the last binary expression handled by the
  /// BinaryExpressionAcc. This information is needed to know where to attach the simplified subtree to in case that
  /// the operator changes.
  /// \param binaryExprNode The last handled binary expression node.
  void setLastVisitedSubtree(AbstractExpr *binaryExprNode) {
    lastVisitedBinaryExp = binaryExprNode;
  }

  /// Sets a new value to firstVisitedBinaryExp that indicates the first binary expression for which operands were
  /// aggregated by the BinaryExpressionAcc. This information is needed when generating the simplified subtree
  /// consisting of the collected operands. If the firstVisitedBinaryExp had any binary expressions, then these will be
  /// attached to the simplified subtree.
  /// \param binaryExprNode The first binary expression for which operands were collected.
  void setFirstVisitedSubtree(AbstractExpr *binaryExprNode) {
    firstVisitedBinaryExp = binaryExprNode;
  }

  /// Clears all the information collected so far in this BinaryExpressionAcc.
  void reset() {
    operands.clear();
    lastVisitedBinaryExp = nullptr;
    numberOfReducedNodes = 0;
  }

  /// Checks whether the accumulation could reduce/simplify any sub-expressions.
  /// \return True if the operands accumulated so far could be simplified by evaluating them, otherwise False.
  bool subtreeIsSimplified() {
    return numberOfReducedNodes > 0;
  }

  /// Checks whether a given operator (opSymbol) is suitable for binary expression accumulation as this approach only
  /// works for commutative operators.
  /// \param opSymbol The operator symbol to be checked for suitability for binary expression accumulation.
  /// \return True if this operator supports binary expression accumulation, otherwise False.
  static bool isSupportedOperator(OpSymbolVariant opSymbol) {
    // all commutative operators
    static const std::vector<LogCompOp> arithmeticOps =
        {LogCompOp::logicalAnd, LogCompOp::logicalOr, LogCompOp::logicalXor};
    static const std::vector<ArithmeticOp> logicalOps =
        {ArithmeticOp::addition, ArithmeticOp::multiplication};

    // accumulator approach works for commutative operators only
    if (std::holds_alternative<ArithmeticOp>(opSymbol)) {  // arithmetic operators
      return std::find(arithmeticOps.begin(), arithmeticOps.end(), std::get<ArithmeticOp>(opSymbol))
          !=arithmeticOps.end();
    } else if (std::holds_alternative<LogCompOp>(opSymbol)) {  // logical operators
      return std::find(logicalOps.begin(), logicalOps.end(), std::get<LogCompOp>(opSymbol))
          !=logicalOps.end();
    }
    return false;
  }

  /// Takes the operands collected so far, including the simplified ones, and creates a multiplicative depth-balanced
  /// tree that can be used to simplify the respective subtree of the AST.
  /// \return A multiplicative depth-balanced tree representing the simplified expression.
  AbstractNode *getSimplifiedSubtree() {
    // check if the node where we started collecting operands (i.e., "lower bound" of simplified tree segment) that has
    // a child that we need to attach to the simplified segment
    for (auto &c : firstVisitedBinaryExp->getChildrenNonNull()) {
      if (dynamic_cast<AbstractBinaryExpr *>(c)!=nullptr) {
        operands.push_back(c->castTo<AbstractExpr>());
      }
    }
    return createMultDepthBalancedTreeFromInputs(operands, operatorSymbol);
  }

  /// Extracts all AbstractLiterals of the operands vector, combines them using the current operator (see
  /// operatorSymbol), and applies the operator to the operands. Adds the result back to the operands vector.
  /// Also keeps track of the number of operands that could be removed using this partial evaluation approach.
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

  /// Adds new operands to the vector of collected operands and triggers evaluation of AbstractLiterals (see method
  /// evaluateLiterals(...)).
  /// \param operandsToBeAdded The operands to be added to the operands vector.
  void addOperands(std::vector<AbstractExpr *> operandsToBeAdded) {
    // we are traversing from leaf to root, hence don't add any binary expressions as they by itself already added
    // their operands to BinaryExpressionAcc
    auto it = std::remove_if(operandsToBeAdded.begin(), operandsToBeAdded.end(), [&](AbstractExpr *ae) {
      return dynamic_cast<AbstractBinaryExpr *>(ae)!=nullptr;
    });
    // insert all remaining operands (of type Variable or AbstractLiteral)
    operands.insert(operands.end(), operandsToBeAdded.begin(), it);
    evaluateLiterals();
  }

  /// Removes all collected operands and replaces the operator symbol by the given one (newSymbol).
  /// \param newSymbol The new operator symbol to be set.
  void removeOperandsAndSetNewSymbol(OpSymbolVariant newSymbol) {
    operands.clear();
    operatorSymbol = newSymbol;
  }

  /// Checks if there are any collected operands.
  /// \return True if any operands were collected so far.
  bool containsOperands() {
    return !operands.empty();
  }

  /// Get the operator symbol of the collected operands.
  /// \return The operator symbol as OpSymbolVariant.
  [[nodiscard]] const OpSymbolVariant &getOperatorSymbol() const {
    return operatorSymbol;
  }
};

class CompileTimeExpressionSimplifier : public Visitor {
 private:
  /// A EvaluationVisitor instance that is used to evaluate parts of the AST in order to simplify them.
  EvaluationVisitor evalVisitor;

 public:
  CompileTimeExpressionSimplifier();

  /// Contains all nodes that can potentially be removed. The decision of a node's removal is to be made by its parent
  /// statement. If this decision is made (i.e., at the end of processing the statement), the node must be deleted from
  /// removableNodes.
  /// - AbstractNode*: A reference to the removable node.
  std::unordered_set<AbstractNode *> removableNodes;

  /// Stores the latest value of a variable while traversing through the AST.
  /// - std::string: The variable's identifier.
  /// - AbstractExpr*: The variable's value.
  std::unordered_map<std::string, AbstractExpr *> variableValues;

  /// Contains pointer to those nodes for which full or partial evaluation could be performed and hence can be deleted
  /// at the end of this simplification traversal.
  /// For example, the expression ArithmeticExpr(LiteralInt(12), OpSymb::add, LiteralInt(42)) will be evaluated to 12+42=54.
  /// The node ArithmeticExpr (and all of its children) will be deleted and replaced by a new node LiteralInt(54).
  std::deque<AbstractNode *> nodesQueuedForDeletion;

  /// An instance to the BinaryExpressionAcc that is needed to simplify nested binary expressions.
  BinaryExpressionAcc binaryExpressionAccumulator;

  /** @defgroup visit Methods implementing the logic of the visitor for each node type.
  *  @{
  */

  void visit(AbstractExpr &elem) override;
  void visit(AbstractNode &elem) override;
  void visit(AbstractStatement &elem) override;
  void visit(ArithmeticExpr &elem) override;
  void visit(Ast &elem) override;
  void visit(Block &elem) override;
  void visit(Call &elem) override;
  void visit(CallExternal &elem) override;
  void visit(Datatype &elem) override;
  void visit(Function &elem) override;
  void visit(FunctionParameter &elem) override;
  void visit(If &elem) override;
  void visit(LiteralBool &elem) override;
  void visit(LiteralFloat &elem) override;
  void visit(LiteralInt &elem) override;
  void visit(LiteralString &elem) override;
  void visit(LogicalExpr &elem) override;
  void visit(Operator &elem) override;
  void visit(ParameterList &elem) override;
  void visit(Return &elem) override;
  void visit(UnaryExpr &elem) override;
  void visit(VarAssignm &elem) override;
  void visit(VarDecl &elem) override;
  void visit(Variable &elem) override;
  void visit(While &elem) override;

  /** @} */ // End of visit group

  /// Checks whether there is a known value for this node. A value can either be an AbstractLiteral or an AbstractExpr
  /// with unknown variables (e.g., function parameters).
  /// \param node The node for which the existence of a value should be determined.
  /// \return True if the node's value is known, otherwise False.
  bool valueIsKnown(AbstractNode *node);

  /// Marks a node as a candidate for deletion. The node's first ancestor that is a statement has to decide whether its
  /// children that are marked to be removed can be deleted or not.
  /// \param node The node for which the evaluation result should be stored.
  void markNodeAsRemovable(AbstractNode *node);

  /// Returns the (first) known value of a node, i.e., the first element of the result vector. If a node has multiple
  /// values (e.g., Return statements), this method will return an exception.
  /// \param node The node for which the first value should be retrieved.
  /// \return The node's value.
  AbstractExpr *getFirstValue(AbstractNode *node);

  /// Evaluates a subtree, i.e., a node and all of its children by using the EvaluationVisitor.
  /// \param node The subtree's root node to be evaluated.
  /// \param valuesOfVariables The variable values required to evaluate this subtree.
  /// \return The evaluation's result as a vector of AbstractLiteral pointers. If the subtree does not include a Return
  ///         statement, then the result vector should always have one element only.
  std::vector<AbstractLiteral *> evaluateNodeRecursive(AbstractNode *node,
                                                       std::unordered_map<std::string,
                                                                          AbstractLiteral *> valuesOfVariables);

  /// This method generalizes the handling of binary expressions, i.e., ArithmeticExpr and LogicalExpr for simplifying
  /// them. If both operand values are known, then the method evaluates them. Otherwise, if the binary expression's
  /// operand is supported for accumulation, the operands are aggregated to possibly simplify them (this only works if
  /// this binary expression is a nested one and there are multiple literals on which the operator can be applied on).
  /// \param arithmeticExpr The arithmetic expression visited.
  void handleBinaryExpression(AbstractBinaryExpr &arithmeticExpr);

  /// Takes the variableValues map and creates a new map containing a copy of all AbstractLiteral values. This map
  /// can then be passed to the EvaluationVisitor to evaluate a given subtree. The original variableValues map, however,
  /// contains AbstractExprs of non-evaluable variables too, for example, x=y+7 where y is a function parameter.
  /// \return A map of (variable identifier, variable value) pairs where variable values are AbstractLiterals.
  std::unordered_map<std::string, AbstractLiteral *> getTransformedVariableMap();

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
  /// \param falseValue The value to be used for the case that the condition evaluates to False.
  /// \return A arithmetic expression of the form condition*trueValue + (1-b)*falseValue.
  static AbstractExpr *generateIfDependentValue(AbstractExpr *condition,
                                                AbstractExpr *trueValue,
                                                AbstractExpr *falseValue);

  /// This method must be called at the end of the visit(...) of each statement. The method makes sure that the binary
  /// expression accumulator is reset and if specified by enqueueStatementForDeletion=True, also marks the given
  /// statement for deletion.
  /// \param statement The statement for which visit(...) was executed.
  /// \param enqueueStatementForDeletion True if this statement should be marked for deletion, otherwise False
  /// (default).
  void cleanUpAfterStatementVisited(AbstractNode *statement, bool enqueueStatementForDeletion = false);

  /// Adds a new (variable identifier, variable value) pair to the map of variable values.
  /// \param variableIdentifier The variable identifier ("name" of the variable).
  /// \param valueAnyLiteralOrAbstractExpr The variable's value. This can be any kind of AbstractLiteral or
  void addVariableValue(const std::string &variableIdentifier, AbstractExpr *valueAnyLiteralOrAbstractExpr);

  /// Checks whether the given node is queued for deletion. Deletion will be carried out at the end of the traversal.
  /// \param node The node to be checked for deletion.
  /// \return True if this node is enqueued for deletion, otherwise False.
  bool isQueuedForDeletion(const AbstractNode *node);
};

#endif //AST_OPTIMIZER_INCLUDE_VISITOR_COMPILETIMEEXPRESSIONSIMPLIFIER_H_
