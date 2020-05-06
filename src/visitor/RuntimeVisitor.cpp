#include "ast_opt/visitor/RuntimeVisitor.h"

#include <utility>
#include "ast_opt/visitor/EvaluationVisitor.h"
#include "ast_opt/ast/MatrixElementRef.h"
#include "ast_opt/ast/For.h"
#include "ast_opt/ast/Variable.h"
#include "ast_opt/ast/VarDecl.h"
#include "ast_opt/ast/MatrixAssignm.h"

void RuntimeVisitor::visit(Ast &elem) {
  Visitor::visit(elem);
}

void RuntimeVisitor::visit(For &elem) {
  // execute initializer
  elem.getInitializer()->accept(*ev);

  auto conditionIsTrue = [&]() -> bool {
    // visit the condition's expression
    elem.getCondition()->accept(*ev);
    // get the expression's evaluation result
    auto cond = *dynamic_cast<LiteralBool *>(ev->getResults().front());
    return cond==LiteralBool(true);
  };

  // while condition evaluates true
  while (conditionIsTrue()) {
    // execute loop body
    elem.getStatementToBeExecuted()->accept(*ev);
    elem.getStatementToBeExecuted()->accept(*this);

    // extract index pairs that have been accessed in this loop iteration and store them globally


    // execute update statement
    elem.getUpdateStatement()->accept(*ev);
    std::cout << std::endl;
  }
}

void RuntimeVisitor::visit(MatrixElementRef &elem) {
  Visitor::visit(elem);

  // a helper utility that either returns the value of an already existing LiteralInt (in AST) or performs evaluation
  // and then retrieves the value of the evaluation result (LiteralInt)
  auto determineIndexValue = [&](AbstractExpr *expr) {
    auto rowIdxLiteral = dynamic_cast<LiteralInt *>(expr);
    // if index is a literal: simply return its value
    if (rowIdxLiteral!=nullptr) return rowIdxLiteral->getValue();
    // if row index is not a literal: evaluate expression
    expr->accept(*ev);
    auto evalResult = ev->getResults().front();
    if (auto evalResultAsLInt = dynamic_cast<LiteralInt *>(evalResult)) {
      return evalResultAsLInt->getValue();
    } else {
      throw std::runtime_error("MatrixElementRef row and column indices must evaluate to LiteralInt!");
    }
  };

  // determine value of row and column index
  int rowIdx = determineIndexValue(elem.getRowIndex());
  int colIdx = determineIndexValue(elem.getColumnIndex());

  std::string varIdentifier;
  if (auto var = dynamic_cast<Variable *>(elem.getOperand())) {
    varIdentifier = var->getIdentifier();
  } else {
    throw std::runtime_error("MatrixElementRef does not refer to a Variable. Cannot continue. Aborting.");
  }

  // store accessed index pair (rowIdx, colidx) and associated variable (matrix) globally
  registerMatrixAccess(varIdentifier, rowIdx, colIdx);
}

RuntimeVisitor::RuntimeVisitor(std::unordered_map<std::string, AbstractLiteral *> funcCallParameterValues) {
  ev = new EvaluationVisitor(std::move(funcCallParameterValues));
}

void RuntimeVisitor::registerMatrixAccess(std::string variableIdentifier, int rowIndex, int columnIndex) {
  std::stringstream ss;
  ss << variableIdentifier << "[" << rowIndex << "]"
     << "[" << columnIndex << "]" << "\t(" << currentMatrixAccessMode << ")";
  variableAccessMap[variableIdentifier][std::pair(rowIndex, columnIndex)] = currentMatrixAccessMode;
  std::cout << ss.str() << std::endl;
}

void RuntimeVisitor::visit(VarDecl &elem) {
  elem.accept(*ev);
}

void RuntimeVisitor::visit(MatrixAssignm &elem) {
  // inline Visitor::visit(elem);
  Visitor::addStatementToScope(elem);
  currentMatrixAccessMode = WRITE;
  elem.getAssignmTarget()->accept(*this);
  currentMatrixAccessMode = READ;
  elem.getValue()->accept(*this);
}

