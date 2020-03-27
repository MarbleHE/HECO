#include <utility>
#include "EvaluationVisitor.h"
#include "AbstractNode.h"
#include "AbstractExpr.h"
#include "AbstractStatement.h"
#include "ArithmeticExpr.h"
#include "Block.h"
#include "Call.h"
#include "CallExternal.h"
#include "Function.h"
#include "If.h"
#include "LiteralBool.h"
#include "LiteralInt.h"
#include "LiteralFloat.h"
#include "LogicalExpr.h"
#include "Operator.h"
#include "Return.h"
#include "Scope.h"
#include "UnaryExpr.h"
#include "VarAssignm.h"
#include "While.h"
#include "GetMatrixElement.h"
#include "Rotate.h"
#include "For.h"
#include "Transpose.h"

EvaluationVisitor::EvaluationVisitor(std::unordered_map<std::string, AbstractLiteral *> funcCallParameterValues)
    : variableValuesForEvaluation(std::move(funcCallParameterValues)) {
}

EvaluationVisitor::EvaluationVisitor() = default;

void EvaluationVisitor::visit(AbstractNode &elem) {
  results.push(std::vector<AbstractLiteral *>());
}

void EvaluationVisitor::visit(AbstractExpr &elem) {
  Visitor::visit(elem);
}

void EvaluationVisitor::visit(AbstractStatement &elem) {
  Visitor::visit(elem);
}

void EvaluationVisitor::visit(ArithmeticExpr &elem) {
  // we first need to evaluate the left- and right-handside as they can consists of nested arithmetic expressions
  elem.getLeft()->accept(*this);
  auto l = getOnlyEvaluationResult(results.top());
  results.pop();
  elem.getRight()->accept(*this);
  auto r = getOnlyEvaluationResult(results.top());
  results.pop();
  results.push({elem.getOperator()->applyOperator(l, r)});
}

void EvaluationVisitor::visit(Block &elem) {
  // a block statement itself does not return anything - its contained statements are just being executed
  for (auto &stmt : elem.getStatements()) {
    stmt->accept(*this);
  }
}

void EvaluationVisitor::visit(Call &elem) {
  // validation: make sure that both Call and Function have the same number of arguments
  if (elem.getArguments().size()!=elem.getFunc()->getParameters().size()) {
    std::stringstream ss;
    ss << "Number of arguments in Call and its called Function does not match (";
    ss << elem.getArguments().size() << " vs. " << elem.getFunc()->getParameters().size();
    ss << ").";
    throw std::logic_error(ss.str());
  }

  // create vector to store parameter values for Function evaluation
  // - std::string stores the variable's identifier
  // - Literal* stores the variable's passed value (as it can be an expression too, we need to evaluate it first)
  std::unordered_map<std::string, AbstractLiteral *> paramValues;

  for (size_t i = 0; i < elem.getFunc()->getParameters().size(); i++) {
    // validation: make sure that datatypes in Call and Function are equal
    auto datatypeCall = *elem.getArguments().at(i)->getDatatype();
    auto datatypeFunc = *elem.getFunc()->getParameters().at(i)->getDatatype();
    if (datatypeCall!=datatypeFunc)
      throw std::logic_error("Datatype in Call and Function mismatch! Cannot continue."
                             "Note: Vector position (index) of parameters in Call and Function must be equal.");

    // variable identifier: retrieve the variable identifier to bind the value to
    auto val = elem.getFunc()->getParameters().at(i)->getValue();
    std::string varIdentifier;
    if (auto var = dynamic_cast<Variable *>(val)) {
      varIdentifier = var->getIdentifier();
    } else {
      throw std::logic_error("FunctionParameter in Call must have a Variable type as value.");
    }

    // variable value: retrieve the variable's value to be passed to the callee
    elem.getArguments().at(i)->getValue()->accept(*this);
    AbstractLiteral *lit = getOnlyEvaluationResult(results.top());
    results.pop();
    // make sure that evaluate returns a Literal
    if (lit==nullptr) throw std::logic_error("There's something wrong! Evaluate should return a single Literal.");

    // store value of lit in vector paramValues with its variable identifier
    // this is to be used to evaluate the Function called by Call
    lit->addLiteralValue(varIdentifier, paramValues);
  }

  // evaluate called function (returns nullptr if function is void)
  Ast subAst(elem.getFunc());

  results.push(subAst.evaluateAst(paramValues, false));
}

void EvaluationVisitor::visit(CallExternal &elem) {
  throw std::runtime_error(
      "evaluateAst(Ast &ast) not implemented for class CallExternal yet! Consider using Call instead.");
}

void EvaluationVisitor::visit(Function &elem) {
  // visit all statements of the Function's body in order of execution
  for (size_t i = 0; i < elem.getBodyStatements().size(); i++) {
    elem.getBodyStatements().at(i)->accept(*this);
  }
}

void EvaluationVisitor::visit(FunctionParameter &elem) {
  Visitor::visit(elem);
}

void EvaluationVisitor::visit(If &elem) {
  elem.getCondition()->accept(*this);
  auto cond = dynamic_cast<LiteralBool *>(getOnlyEvaluationResult(results.top()));
  results.pop();
  if (cond==nullptr)
    throw std::logic_error("Condition in If statement must evaluate to a LiteralBool! Cannot continue.");
  // check which of the branches must be evaluated
  if (*cond==LiteralBool(true) && elem.getThenBranch()!=nullptr) {
    elem.getThenBranch()->accept(*this);
  } else if (elem.getElseBranch()!=nullptr) {
    elem.getElseBranch()->accept(*this);
  }
}

template<typename T, class LiteralClass>
Matrix<T> *EvaluationVisitor::evaluateAbstractExprMatrix(EvaluationVisitor &ev, AbstractMatrix &mx) {
  auto mxDim = mx.getDimensions();
  std::vector<std::vector<T>> evaluatedValues(mxDim.numRows, std::vector<T>(mxDim.numColumns));
  for (int i = 0; i < mxDim.numRows; ++i) {
    for (int j = 0; j < mxDim.numColumns; ++j) {
      mx.getElementAt(i, j)->accept(ev);
      auto result = getOnlyEvaluationResult(results.top());
      results.pop();
      evaluatedValues[i][j] = result->castTo<LiteralClass>()->getValue();
    }
  }
  return new Matrix<T>({evaluatedValues});
}

void EvaluationVisitor::visit(LiteralBool &elem) {
  if (dynamic_cast<Matrix<bool> *>(elem.getMatrix())) {  // if this is a scalar, i.e, Matrix<bool> of single element
    results.push({&elem});
  } else {  // otherwise, this must be a Matrix<AbstractExpr*> -> we need to evaluate each expression
    Matrix<bool> *evaluatedExpressions = evaluateAbstractExprMatrix<bool, LiteralBool>(*this, *elem.getMatrix());
    results.push({new LiteralBool(evaluatedExpressions)});
  }
}

void EvaluationVisitor::visit(LiteralInt &elem) {
  if (dynamic_cast<Matrix<int> *>(elem.getMatrix())) {  // if this is a scalar, i.e, Matrix<int> of single element
    results.push({&elem});
  } else {  // otherwise, this must be a Matrix<AbstractExpr*> -> we need to evaluate each expression
    Matrix<int> *evaluatedExpressions = evaluateAbstractExprMatrix<int, LiteralInt>(*this, *elem.getMatrix());
    results.push({new LiteralInt(evaluatedExpressions)});
  }
}

void EvaluationVisitor::visit(LiteralString &elem) {
  // if this is a scalar, i.e, Matrix<std::string> of single element
  if (dynamic_cast<Matrix<std::string> *>(elem.getMatrix())) {
    results.push({&elem});
  } else {  // otherwise, this must be a Matrix<AbstractExpr*> -> we need to evaluate each expression
    Matrix<std::string>
        *evaluatedExpressions = evaluateAbstractExprMatrix<std::string, LiteralString>(*this, *elem.getMatrix());
    results.push({new LiteralString(evaluatedExpressions)});
  }
}

void EvaluationVisitor::visit(LiteralFloat &elem) {
  if (dynamic_cast<Matrix<float> *>(elem.getMatrix())) {  // if this is a scalar, i.e, Matrix<float> of single element
    results.push({&elem});
  } else {  // otherwise, this must be a Matrix<AbstractExpr*> -> we need to evaluate each expression
    Matrix<float> *evaluatedExpressions = evaluateAbstractExprMatrix<float, LiteralFloat>(*this, *elem.getMatrix());
    results.push({new LiteralFloat(evaluatedExpressions)});
  }
}

void EvaluationVisitor::visit(LogicalExpr &elem) {
  // we first need to evaluate the left and right hand-side as they can consist of nested expressions
  elem.getLeft()->accept(*this);
  auto l = getOnlyEvaluationResult(results.top());
  results.pop();
  elem.getRight()->accept(*this);
  auto r = getOnlyEvaluationResult(results.top());
  results.pop();
  results.push({elem.getOperator()->applyOperator(l, r)});
}

void EvaluationVisitor::visit(Operator &elem) {
  Visitor::visit(elem);
}

void EvaluationVisitor::visit(Return &elem) {
  std::vector<AbstractLiteral *> result;
  for (auto &expr : elem.getReturnExpressions()) {
    expr->accept(*this);
    auto exprEvaluationResult = results.top();
    results.pop();
    result.insert(result.end(), exprEvaluationResult.begin(), exprEvaluationResult.end());
  }
  results.push(result);
}

void EvaluationVisitor::visit(UnaryExpr &elem) {
  elem.getRight()->accept(*this);
  results.push({elem.getOp()->applyOperator(getOnlyEvaluationResult(results.top()))});
}

void EvaluationVisitor::visit(Rotate &elem) {
  // visit the operand: we need to evaluate it as it can be an expression itself, e.g., rotate([1; 2; 3]+[1;9;2], 2)
  elem.getOperand()->accept(*this);
  // visit the rotation factor: we need to evaluate it as it can be an expression itself, e.g., rotate(vec, y+1)
  elem.getRotationFactor()->accept(*this);
  // check whether the rotation factor is known, otherwise we can stop right here because we cannot perform rotation if
  // the rotation factor is unknown
  auto rotationFactor = dynamic_cast<LiteralInt *>(getOnlyEvaluationResult(results.top()));
  results.pop();
  auto operand = getOnlyEvaluationResult(results.top());
  results.pop();
  if (rotationFactor==nullptr || operand==nullptr) return;

  // rotate the Literal in-place by using a copy of the operand
  auto clonedOperand = operand->clone(false)->castTo<AbstractLiteral>();
  clonedOperand->getMatrix()->rotate(rotationFactor->getValue(), true);
  results.push({clonedOperand});
}

void EvaluationVisitor::visit(Transpose &elem) {
  // visit the operand: we need to evaluate it as it can be an expression itself, e.g., transpose([1,2,3]+[2,3,1])
  elem.getOperand()->accept(*this);
  auto operand = getOnlyEvaluationResult(results.top())->clone(false)->castTo<AbstractLiteral>();
  results.pop();
  // rotate the cloned Literal in-place
  operand->getMatrix()->transpose(true);
  results.push({operand});
}

void EvaluationVisitor::visit(VarAssignm &elem) {
  elem.getValue()->accept(*this);
  auto val = getOnlyEvaluationResult(results.top());
  results.pop();
  updateVarValue(elem.getIdentifier(), val);
}

void EvaluationVisitor::visit(VarDecl &elem) {
  if (elem.getInitializer()!=nullptr) {
    elem.getInitializer()->accept(*this);
    // get the variable declaration's evaluated value
    auto value = getOnlyEvaluationResult(results.top());
    results.pop();
    // save the variable's value
    updateVarValue(elem.getIdentifier(), value);
    results.push({value});
  } else {
    updateVarValue(elem.getIdentifier(), nullptr);
    results.push({});
  }
}

void EvaluationVisitor::visit(Variable &elem) {
  results.push({getVarValue(elem.getIdentifier())});
}

void EvaluationVisitor::visit(While &elem) {
  auto conditionIsTrue = [&]() -> bool {
    // visit the condition's expression
    elem.getCondition()->accept(*this);
    // get the expression's evaluation result
    auto cond = *dynamic_cast<LiteralBool *>(getOnlyEvaluationResult(results.top()));
    results.pop();
    return cond==LiteralBool(true);
  };

  // if and as long as the condition is true, repeat the following
  while (conditionIsTrue()) {
    // execute the While-loop's body
    elem.getBody()->accept(*this);
  }
}

void EvaluationVisitor::visit(For &elem) {
  auto checkCondition = [&]() -> bool {
    // visit the condition's expression
    elem.getCondition()->accept(*this);
    // get the expression's evaluation result
    auto cond = *dynamic_cast<LiteralBool *>(getOnlyEvaluationResult(results.top()));
    results.pop();
    return cond==LiteralBool(true);
  };

  for (elem.getInitializer()->accept(*this); checkCondition(); elem.getUpdateStatement()->accept(*this)) {
    elem.getStatementToBeExecuted()->accept(*this);
  }
}

void EvaluationVisitor::visit(GetMatrixElement &elem) {
  // operand
  elem.getOperand()->accept(*this);
  auto operand = dynamic_cast<AbstractLiteral *>(getOnlyEvaluationResult(results.top()));
  results.pop();
  // row index
  elem.getRowIndex()->accept(*this);
  auto rowIdx = dynamic_cast<LiteralInt *>(getOnlyEvaluationResult(results.top()));
  results.pop();
  // column index
  elem.getColumnIndex()->accept(*this);
  auto columnIdx = dynamic_cast<LiteralInt *>(getOnlyEvaluationResult(results.top()));
  results.pop();
  // retrieve and store the specified element
  auto matrixElement = operand->getMatrix()->getElementAt(rowIdx->getValue(), columnIdx->getValue());
  results.push({matrixElement->castTo<AbstractLiteral>()});
}

void EvaluationVisitor::visit(Ast &elem) {
  Visitor::visit(elem);
}

const std::vector<AbstractLiteral *> &EvaluationVisitor::getResults() {
  std::vector<AbstractLiteral *> *resultValues = &results.top();
  if (!flagPrintResult) return *resultValues;
  // print result if flag 'printResult' is set
  if (resultValues->empty()) {
    std::cout << "void" << std::endl;
  } else {
    std::stringstream outStr;
    for (auto &resultLiteral : *resultValues) outStr << resultLiteral->toString(false) << std::endl;
    std::cout << outStr.str();
  }
  return *resultValues;
}

AbstractLiteral *EvaluationVisitor::getOnlyEvaluationResult(std::vector<AbstractLiteral *> evaluationResult) {
  if (evaluationResult.size() > 1) {
    throw std::logic_error("Unexpected number of results returned (1 vs. "
                               + std::to_string(evaluationResult.size()) + ").");
  }
  return evaluationResult.front();
}

AbstractLiteral *EvaluationVisitor::getVarValue(const std::string &variableIdentifier) {
  auto it = variableValuesForEvaluation.find(variableIdentifier);
  if (it==variableValuesForEvaluation.end())
    throw std::logic_error("Trying to retrieve value for variable not declared yet: " + variableIdentifier);
  return it->second;
}

void EvaluationVisitor::updateVarValue(const std::string &variableIdentifier, AbstractLiteral *newValue) {
  // use the bracket [] operator to silently overwrite any existing variable value
  variableValuesForEvaluation[variableIdentifier] = newValue;
}

void EvaluationVisitor::updateVarValues(std::unordered_map<std::string, AbstractLiteral *> variableValues) {
  std::for_each(variableValues.begin(),
                variableValues.end(),
                [this](const std::pair<std::string, AbstractLiteral *> &mapEntry) {
                  this->updateVarValue(mapEntry.first, mapEntry.second);
                });
}

void EvaluationVisitor::setFlagPrintResult(bool printResult) {
  EvaluationVisitor::flagPrintResult = printResult;
}

void EvaluationVisitor::reset() {
  variableValuesForEvaluation.clear();
}
