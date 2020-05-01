#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_EVALUATIONVISITOR_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_EVALUATIONVISITOR_H_

#include <vector>
#include <stack>
#include <unordered_map>
#include <string>
#include "Visitor.h"
#include "ast_opt/ast/MatrixElementRef.h"
#include "ast_opt/ast/Matrix.h"

class EvaluationVisitor : public Visitor {
 private:
  typedef std::vector<AbstractLiteral *> result_t;

  /// Determines whether getResults should also print the results to stdout.
  bool flagPrintResult{false};

  /// Stores intermediate evaluation results. At the end of the evaluation, results.top() contains the value of the
  /// return statement.
  std::stack<result_t> results = {};

  /// Tests whether a given evaluation result only contains a single element. This is used to verify that certain
  /// expressions only return a single element.
  /// \param evaluationResult The vector of AbstractLiteral pointers to be tested.
  /// \return The single AbstractLiteral pointer, otherwise throws an exception.
  /// \throws std::logic_error if more than one element is in the vector of AbstractLiteral pointers.
  static AbstractLiteral *getOnlyEvaluationResult(std::vector<AbstractLiteral *> evaluationResult);

  /// This map stores the variables values and serves as lookup table and central storage during the evaluation process.
  /// - std::string: The variable identifier.
  /// - AbstractLiteral*: The evaluated value of the variable.
  std::unordered_map<std::string, AbstractLiteral *> variableValuesForEvaluation;

  /// Returns the value in variableValuesForEvaluation for a given variable identifier.
  /// \param variableIdentifier The identifier of which the value should be determined.
  /// \return The variable's value.
  AbstractLiteral *getVarValue(const std::string &variableIdentifier);

  /// Updates the value of the variable identified by a given identifier variableIdentifier, by a new value (newValue).
  /// \param variableIdentifier The variable to be updated.
  /// \param newValue The new value to be assigned to the variable.
  void updateVarValue(const std::string &variableIdentifier, AbstractLiteral *newValue);

 public:
  explicit EvaluationVisitor(std::unordered_map<std::string, AbstractLiteral *> funcCallParameterValues);

  EvaluationVisitor();

  void visit(AbstractExpr &elem) override;

  void visit(AbstractNode &elem) override;

  void visit(AbstractStatement &elem) override;

  void visit(ArithmeticExpr &elem) override;

  void visit(Ast &elem) override;

  void visit(Block &elem) override;

  void visit(Call &elem) override;

  void visit(CallExternal &elem) override;

  void visit(For &elem) override;

  void visit(Function &elem) override;

  void visit(FunctionParameter &elem) override;

  void visit(If &elem) override;

  void visit(LiteralBool &elem) override;

  void visit(LiteralFloat &elem) override;

  void visit(LiteralInt &elem) override;

  void visit(LiteralString &elem) override;

  void visit(LogicalExpr &elem) override;

  void visit(Operator &elem) override;

  void visit(Return &elem) override;

  void visit(OperatorExpr &elem) override;

  void visit(Rotate &elem) override;

  void visit(MatrixElementRef &elem) override;

  void visit(MatrixAssignm &elem) override;

  void visit(Transpose &elem) override;

  void visit(UnaryExpr &elem) override;

  void visit(VarAssignm &elem) override;

  void visit(VarDecl &elem) override;

  void visit(Variable &elem) override;

  void visit(While &elem) override;

  void visit(GetMatrixSize &elem) override;

  const std::vector<AbstractLiteral *> &getResults();

  void setFlagPrintResult(bool printResult);

  void reset();

  void updateVarValues(std::unordered_map<std::string, AbstractLiteral *> variableValues);

  template<typename T, typename U>
  Matrix<T> *evaluateAbstractExprMatrix(EvaluationVisitor &ev, AbstractMatrix &mx);
};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_EVALUATIONVISITOR_H_
