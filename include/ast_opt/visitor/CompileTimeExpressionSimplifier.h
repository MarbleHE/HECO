#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_COMPILETIMEEXPRESSIONSIMPLIFIER_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_COMPILETIMEEXPRESSIONSIMPLIFIER_H_

#include <string>
#include <vector>
#include <deque>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <set>
#include <utility>
#include "ast_opt/utilities/NodeUtils.h"
#include "ast_opt/visitor/Visitor.h"
#include "ast_opt/visitor/EvaluationVisitor.h"
#include "ast_opt/ast/ArithmeticExpr.h"
#include "ast_opt/ast/LiteralFloat.h"
#include "ast_opt/ast/Variable.h"
#include "ast_opt/ast/OperatorExpr.h"
#include "ast_opt/utilities/Scope.h"



/**
 * A helper struct that is used by emittedVariableDeclarations and helps to keep track of the relationship between a
 * variable (given as pair of identifier an scope), the associated (emitted) variable declaration statement, and a
 * reference to all emitted variable assignments that depend on this variable declaration. This allows to determine
 * at the end of the traversal if the emitted VarAssignm was meanwhile deleted and we do not need the VarDecl anymore.
 */
struct EmittedVariableData {
  /// the emitted variable declaration statement
  AbstractNode *varDeclStatement;
  /// the emitted variable assignment statements that require this declaration
  std::unordered_set<AbstractNode *> emittedVarAssignms;
  /// other nodes (e.g., MatrixAssignm) that depend on this variable declaration
  std::unordered_set<AbstractNode *> dependentAssignms;

 public:
  explicit EmittedVariableData(AbstractNode *varDeclStatement) : varDeclStatement(varDeclStatement) {}

  void addVarAssignm(AbstractNode *varAssignm) { emittedVarAssignms.insert(varAssignm); }

  void removeVarAssignm(AbstractNode *varAssignm) { emittedVarAssignms.erase(varAssignm); }

  bool hasNoReferringAssignments() { return emittedVarAssignms.empty() && dependentAssignms.empty(); }

  AbstractNode *getVarDeclStatement() { return varDeclStatement; }
};

/**
 * A struct containing all configuration parameters that can be passed to the CompileTimeExpressionSimplifier.
 */
struct CtesConfiguration {
  CtesConfiguration() = default;

  explicit CtesConfiguration(int maxNumLoopUnrollings) : maxNumLoopUnrollings(maxNumLoopUnrollings) {}

  /// A counter that indicates how many loops have been fully or partially unrolled so far.
  int numUnrolledLoopsCounter = 0;

  /// Indicates the total number of allowed loop unrolling operations. If the maximum is reached, any further For-loops
  /// are simplified but not fully or partially unrolled.
  const int maxNumLoopUnrollings = -1;

  /// Specifies up to which number of iterations a loop is fully unrolled.
  const int fullyUnrollLoopMaxNumIterations = 512;

  /// Specifies the number of ciphertext slots that partial loop unrolling optimizes for.
  const int partiallyUnrollLoopNumCipherslots = 3;

  /// Determines whether the number of total allowed loop unrollings is already exhausted.
  /// \return True if the maxNumLoopUnrollings is not reached yet or if there is no limitation in the number of total
  /// loop unrollings. Otherwise returns False.
  bool isUnrollLoopAllowed() {
    return allowsInfiniteLoopUnrollings() || (numUnrolledLoopsCounter < maxNumLoopUnrollings);
  }

  /// Determines whether there is any limitation in the number of loop unrolling operations.
  /// \return True if there is no restriction w.r.t. number of loop unrollings.
  bool allowsInfiniteLoopUnrollings() {
    return maxNumLoopUnrollings==-1;
  }
};

class CompileTimeExpressionSimplifier : public Visitor {
 private:
  /// A CompileTimeExpressionSimplifier configuration object containing configuration parameters.
  CtesConfiguration ctes;

  /// A EvaluationVisitor instance that is used to evaluate parts of the AST in order to simplify them.
  EvaluationVisitor evalVisitor;

  /// Keeps track of all emitted variable declarations and maps each to an associated EmittedVariableData pointer.
  std::map<ScopedVariable, EmittedVariableData *> emittedVariableDeclarations;

  /// Maps emitted VarAssignms to their corresponding VarDecl statement in emittedVariableDeclarations.
  std::map<AbstractNode *,
           std::map<ScopedVariable, EmittedVariableData *>::iterator> emittedVariableAssignms;

  /// A counter that keeps track of the nesting level while visiting For-loops. The first value indicates the
  /// depth of the currently visiting loop body. The second value the depth of the deepest loop. For example:
  ///   ...
  ///   for (...) {     // currentLoopDepth_maxLoopDepth = (1,1)
  ///      ...
  ///      for (...) {   // currentLoopDepth_maxLoopDepth = (2,2)
  ///         ...
  ///      }             // currentLoopDepth_maxLoopDepth = (1,2)
  ///   }                // currentLoopDepth_maxLoopDepth = (0,0)
  ///   ...
  std::pair<int, int> currentLoopDepth_maxLoopDepth = {std::pair(0, 0)};

  /// A method to be called immediately after entering the For-loop's visit method.
  /// updates currentLoopDepth_maxLoopDepth
  void enteredForLoop();

  /// A method to be called before leaving the For-loop's visit method.
  /// updates currentLoopDepth_maxLoopDepth
  void leftForLoop();

 public:
  CompileTimeExpressionSimplifier();

  explicit CompileTimeExpressionSimplifier(CtesConfiguration cfg);

  /// Stores the latest value of a variable while traversing through the AST. Entries in this map consist of a key
  /// (pair) that is made of a variable identifier (first) and the scope where the variable was declared in (second).
  /// The entry of the variableValues map is the current value of the associated variable.
  VariableValuesMap variableValues;

  /// Contains pointer to those nodes for which full or partial evaluation could be performed and hence can be deleted
  /// at the end of this simplification traversal.
  /// For example, the arithmetic expression represented by ArithmeticExpr( LiteralInt(12), OpSymb::add, LiteralInt(42))
  /// will be evaluated to 12+42=54. The node ArithmeticExpr (and all of its children) will be deleted and replaced
  /// by a new node LiteralInt(54).
  std::deque<AbstractNode *> nodesQueuedForDeletion;

  /** @defgroup visit Methods implementing the logic of the visitor for each node type.
  *  @{
  */
  void visit(AbstractExpr &elem) override;

  void visit(AbstractNode &elem) override;

  void visit(AbstractStatement &elem) override;

  void visit(ArithmeticExpr &elem) override;

  void visit(AbstractMatrix &elem) override;

  /// Simplify entire AST
  /// Will reset any scope/variable information that the visitor might have held from previous calls
  void visit(Ast &elem) override;

  void visit(Block &elem) override;

  void visit(Call &elem) override;

  void visit(CallExternal &elem) override;

  void visit(Datatype &elem) override;

  void visit(Function &elem) override;

  void visit(FunctionParameter &elem) override;

  void visit(For &elem) override;

  void visit(MatrixElementRef &elem) override;

  void visit(GetMatrixSize &elem) override;

  void visit(If &elem) override;

  void visit(LiteralBool &elem) override;

  void visit(LiteralFloat &elem) override;

  void visit(LiteralInt &elem) override;

  void visit(LiteralString &elem) override;

  void visit(LogicalExpr &elem) override;

  void visit(Operator &elem) override;

  void visit(OperatorExpr &elem) override;

  void visit(ParameterList &elem) override;

  void visit(Return &elem) override;

  void visit(UnaryExpr &elem) override;

  void visit(VarAssignm &elem) override;

  void visit(VarDecl &elem) override;

  void visit(Variable &elem) override;

  void visit(While &elem) override;

  void visit(Rotate &elem) override;

  void visit(MatrixAssignm &elem) override;

  void visit(Transpose &elem) override;
  /** @} */ // End of visit group

  /// Checks whether the given node has a known value. A value is considered as known if
  ///  - the node itself is any subtype of an AbstractLiteral this includes matrices of a concrete type (e.g.,
  ///    LiteralInt containing Matrix<int>) but not matrices containing AbstractExprs,
  ///  - the node is a Variable and the referred value is known
  ///  - or the node's value is not relevant anymore as it is marked for deletion.
  /// \param node The node for which the presence of a value should be determined.
  /// \return True if the node's value is known, otherwise False.
  bool hasKnownValue(AbstractNode *node);

  /// Returns the value of the given node that is either the node itself if the node is an subtype of AbstractLiteral
  /// or the known value (AbstractExpr) stored previously in the variableValues map.
  /// \param node The node for which the value should be retrieved.
  /// \throws std::invalid_argument exception if the node does not have a known value. Must be checked before using the
  ///         hasKnownValue method.
  /// \return The node's value as AbstractExpr.
  AbstractExpr *getKnownValue(AbstractNode *node);

  /// Evaluates a subtree, i.e., a node and all of its children by using the EvaluationVisitor.
  /// \param node The subtree's root node to be evaluated.
  /// \param valuesOfVariables The variable values required to evaluate this subtree.
  /// \return The evaluation's result as a vector of AbstractLiteral pointers. If the subtree does not include a Return
  ///         statement, then the result vector should always have one element only.
  std::vector<AbstractLiteral *> evaluateNodeRecursive(
      AbstractNode *node, std::unordered_map<std::string, AbstractLiteral *> valuesOfVariables);

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
  /// \return An arithmetic expression of the form condition*trueValue + (1-b)*falseValue.
  static AbstractExpr *generateIfDependentValue(
      AbstractExpr *condition, AbstractExpr *trueValue, AbstractExpr *falseValue);


  /// Checks whether the given node is queued for deletion. Deletion will be carried out at the end of the traversal.
  /// \param node The node to be checked for deletion.
  /// \return True if this node is enqueued for deletion, otherwise False.
  bool isQueuedForDeletion(const AbstractNode *node);



  /// Takes an OperatorExpr consisting of a logical operator (i.e., AND, XOR, OR) and applies the Boolean laws to
  /// simplify the expression. For example, the expression <anything> AND False always evaluates to False, hence we can
  /// replace this OperatorExpr by the boolean value (LiteralBool) False. Other considered rules include:
  ///   * <anything> AND False ⟹ False
  ///   * <anything> AND True  ⟹ <anything>
  ///   * <anything> OR True   ⟹ True
  ///   * <anything> OR False  ⟹ <anything>
  ///   * <anything> XOR False ⟹ <anything>
  ///   * <anything> XOR True  ⟹ !<anything>  [not implemented yet]
  /// where <anything> denotes an arbitrary logical expression of the same logical operator.
  /// \param elem The OperatorExpr that should be simplified using Boolean laws.
  static void simplifyLogicalExpr(OperatorExpr &elem);

  /// Identify all variables that are written and read from in the body + condition of a for-loop
  /// \param forLoop The for loop to investigate
  /// \param VariableValues Initial VariableValues BEFORE the block
  /// \return Variables, with their associated scopes, that match the criteria
  std::set<ScopedVariable> identifyReadWriteVariables(For &forLoop, VariableValuesMap VariableValues);

  /// Creates a new VarAssignm statement of the variable that the given iterator (variableToEmit) is pointing to.
  /// The method ensures that there exists a variable declaration statement (VarDecl) in the scope where this
  /// variable was originally declared.
  /// Returns a set, because emitting a variable might require emitting temporary variables at the same time
  /// \param variableToEmit The variable to be emitted, i.e., for that a variable assignment statement should be
  /// generated.
  /// \return A  set of variable assignment statements for the given variable (variableToEmit).
  std::set<VarAssignm *> emitVariableAssignment(ScopedVariable variableToEmit);

  /// Emit variable assignments (and/or declaration, if necessary) for each of variables
  /// The VarDecl will be created in the scope associated with the variable
  /// \param variables  Variables for which VarDecls are to be emitted, and their associated scope
  /// \return Set of variable assignment statements for the given variables.
  std::set<VarAssignm *> emitVariableAssignments(std::set<ScopedVariable> variables);

  /// Creates a new VarDecl statements of the variable that the given iterator (variableToEmit) is pointing to.
  /// The variable declaration is emitted as the first statement in the scope where the variable was initially
  /// declared. The generated declaration statement is added to the emittedVariableDeclarations map to keep track of
  /// it. On contrary to emitVariableAssignment, this method automatically adds the statement to the AST instead of
  /// returning the generated statement.
  /// \param variableToEmit The variable to be emitted, i.e., for that a variable declaration statement should be
  /// generated.
  void emitVariableDeclaration(ScopedVariable variableToEmit);

  /// Marks the given node for deletion. The node will be deleted after all nodes of the AST have been visited.
  /// \param node The node to be marked for deletion.
  void enqueueNodeForDeletion(AbstractNode *node);

  /// Sets a new value matrixElementValue to the position indicated by (row, column) in matrix referred by
  /// variableIdentifier. This implements variable assignments of the form M[rowIdx][colIdx] = value; where value is
  /// a single element (AbstractExpr or primitive, e.g., int).
  /// \param variableIdentifier A variable identifier that must refer to a matrix, i.e., any subtype of an
  /// AbstractLiteral.
  /// \param row The row index where the new value should be written to.
  /// \param column The column index where the new value should be written to.
  /// \param matrixElementValue The matrix value that should be written to the index given as (row, column).
  void setMatrixVariableValue(const std::string &variableIdentifier,
                              int row,
                              int column,
                              AbstractExpr *matrixElementValue);

  /// Appends a row/column to a matrix or overwrites an existing row/column. This implements variable assignments of
  /// the form M[idx] = vec; where vec is either a row vector, e.g., [4 2 1] or a column vector, e.g., [4; 2; 1].
  /// \param variableIdentifier A variable identifier that must refer to a matrix, i.e., any subtype of an
  /// AbstractLiteral.
  /// \param posIndex The index where the row/column should be appended to. If matrixRowOrColumn is a row vector,
  /// this index is considered as row index. Otherwise, if matrixRowOrColumn is a column vector, this index is
  /// considered as column index.
  /// \param matrixRowOrColumn An AbstractLiteral consisting of a (1,x) or (x,1) matrix.
  void appendVectorToMatrix(const std::string &variableIdentifier, int posIndex, AbstractExpr *matrixRowOrColumn);


  /// Handles the partial loop unrolling to enable batching of the loop's body statements.
  /// \param elem The For-loop to be unrolled.
  /// \return A pointer to the new node if the given For-loop was replaced in the children vector of the For-loop's
  /// parent.
//  AbstractNode *doPartialLoopUnrolling(For &elem);

  /// When a Block has been visited, some of its statements might have been replaced by nullptrs
  /// This function removes those nullptrs from the Block. If the resulting Block is empty,
  /// It also replaces the Block with nullptr in its parent
  void cleanUpBlock(Block &elem);
};

/// Takes a Literal (e.g., LiteralInt) and checks whether its values are defined using a Matrix<AbstractExpr*>. In
/// this case, checks if the value (e.g., of type int) is known of each element such that Matrix<AbstractExpr*> can
/// be replaced by a Matrix<T> (e.g., Matrix<int>) where T is the Literal's associated primitive type (e.g., int).
/// If the matrix could be simplified, the current Literal is replaced by a new Literal of type U containing the
/// simplified matrix.
/// \tparam T The type of the elements of the respective AbstractLiteral subtype, e.g., int, float.
/// \tparam U The AbstractLiteral subtype, e.g., LiteralInt, LiteralFloat.
/// \param elem The AbstractLiteral subtype that should be simplified.
template<typename T, typename U>
void simplifyAbstractExprMatrix(U &elem);

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_COMPILETIMEEXPRESSIONSIMPLIFIER_H_
