#include "ast_opt/visitor/Visitor.h"
#include "ast_opt/visitor/PrintVisitor.h"
#include "ast_opt/ast/Ast.h"
#include "ast_opt/ast/AbstractNode.h"
#include "ast_opt/ast/AbstractExpr.h"
#include "ast_opt/ast/AbstractStatement.h"
#include "ast_opt/ast/ArithmeticExpr.h"
#include "ast_opt/ast/Block.h"
#include "ast_opt/ast/Call.h"
#include "ast_opt/ast/CallExternal.h"
#include "ast_opt/ast/Function.h"
#include "ast_opt/ast/If.h"
#include "ast_opt/ast/LiteralBool.h"
#include "ast_opt/ast/For.h"
#include "ast_opt/ast/LiteralInt.h"
#include "ast_opt/ast/LiteralFloat.h"
#include "ast_opt/ast/LogicalExpr.h"
#include "ast_opt/ast/Operator.h"
#include "ast_opt/ast/Return.h"
#include "ast_opt/ast/UnaryExpr.h"
#include "ast_opt/ast/VarAssignm.h"
#include "ast_opt/ast/While.h"
#include "ast_opt/ast/Rotate.h"
#include "ast_opt/ast/Transpose.h"
#include "ast_opt/ast/MatrixElementRef.h"
#include "ast_opt/ast/OperatorExpr.h"
#include "ast_opt/ast/MatrixAssignm.h"
#include "ast_opt/ast/GetMatrixSize.h"
#include "ast_opt/utilities/Scope.h"

void Visitor::visit(Ast &elem) {
  // assumption: AST is always the enclosing object that points to the root
  stmtToScopeMapper.clear();
  this->curScope = new Scope("global", elem.getRootNode(), nullptr);
  elem.getRootNode()->accept(*this);
}

void Visitor::visit(AbstractNode &elem) {
  auto children = elem.getChildren();
  for (auto &c : children) {
    c->accept(*this);
  }
}

void Visitor::visit(AbstractExpr &elem) {
  auto children = elem.getChildren();
  for (auto &c : children) {
    c->accept(*this);
  }
}

void Visitor::visit(AbstractStatement &elem) {
  auto children = elem.getChildren();
  for (auto &c : children) {
    c->accept(*this);
  }
}

void Visitor::visit(ArithmeticExpr &elem) {
  elem.getLeft()->accept(*this);
  elem.getOperator()->accept(*this);
  elem.getRight()->accept(*this);
}

void Visitor::visit(Block &elem) {
  addStatementToScope(elem);
  changeToInnerScope(elem.getUniqueNodeId(), &elem);
  for (auto &stat : elem.getStatements()) {
    stat->accept(*this);
  }
  changeToOuterScope();
}

void Visitor::visit(Call &elem) {
  // callee
  elem.getFunc()->accept(*this);
  // arguments
  changeToInnerScope(elem.getFunc()->getUniqueNodeId(), elem.getFunc());
  for (auto arg : elem.getArguments()) {
    arg->accept(*this);
  }
  changeToOuterScope();
}

void Visitor::visit(CallExternal &elem) {
  // arguments for calling function
  if (!elem.getArguments().empty()) {
    for (auto &fp : elem.getArguments()) {
      fp->accept(*this);
    }
  }
}

void Visitor::visit(Function &elem) {
  addStatementToScope(elem);
  changeToInnerScope(elem.getUniqueNodeId(), &elem);
  // visit FunctionParameter
  if (auto fp = elem.getParameterList()) fp->accept(*this);
  // visit Body statements
  if (auto body = elem.getBody()) body->accept(*this);
  changeToOuterScope();
}

void Visitor::visit(FunctionParameter &elem) {
  elem.getDatatype()->accept(*this);
  elem.getValue()->accept(*this);
}

void Visitor::visit(If &elem) {
  addStatementToScope(elem);

  // condition
  elem.getCondition()->accept(*this);

  // if the user supplied only a single AbstractStatement as then- or else-branch, instead of a Block
  // consisting of multiple commands, we need in the following manually open a new scope

  // thenBranch
  if (elem.getThenBranch()!=nullptr) {
    changeToInnerScope(elem.getThenBranch()->getUniqueNodeId(), &elem);
    elem.getThenBranch()->accept(*this);
    changeToOuterScope();
  }

  // elseBranch
  if (elem.getElseBranch()!=nullptr) {
    changeToInnerScope(elem.getElseBranch()->getUniqueNodeId(), &elem);
    elem.getElseBranch()->accept(*this);
    changeToOuterScope();
  }
}

void Visitor::visit(LiteralBool &elem) {
  // If this is not a matrix containing primitives, like Matrix<int>, Matrix<float>, etc. then this must be a
  // Matrix<AbstractExpr*> which requires visiting its elements.
  if (dynamic_cast<Matrix<AbstractExpr *> *>(elem.getMatrix())!=nullptr) {
    elem.getMatrix()->accept(*this);
  }
}

void Visitor::visit(LiteralInt &elem) {
  // If this is not a matrix containing primitives, like Matrix<int>, Matrix<float>, etc. then this must be a
  // Matrix<AbstractExpr*> which requires visiting its elements.
  if (dynamic_cast<Matrix<AbstractExpr *> *>(elem.getMatrix())!=nullptr) {
    elem.getMatrix()->accept(*this);
  }
}

void Visitor::visit(LiteralString &elem) {
  // If this is not a matrix containing primitives, like Matrix<int>, Matrix<float>, etc. then this should be a
  // Matrix<AbstractExpr*> which requires visiting its elements.
  if (dynamic_cast<Matrix<AbstractExpr *> *>(elem.getMatrix())!=nullptr) {
    elem.getMatrix()->accept(*this);
  }
}

void Visitor::visit(LiteralFloat &elem) {
  // If this is not a matrix containing primitives, like Matrix<int>, Matrix<float>, etc. then this should be a
  // Matrix<AbstractExpr*> which requires visiting its elements.
  if (dynamic_cast<Matrix<AbstractExpr *> *>(elem.getMatrix())!=nullptr) {
    elem.getMatrix()->accept(*this);
  }
}

void Visitor::visit(LogicalExpr &elem) {
  // left
  elem.getLeft()->accept(*this);
  // operator
  elem.getOperator()->accept(*this);
  // right
  elem.getRight()->accept(*this);
}

void Visitor::visit(For &elem) {
  // a for-statement
  // e.g., for (int i = 0; i < N; i++) { cout << i << endl; }
  changeToInnerScope(elem.getUniqueNodeId(), &elem);
  // initializer
  if (elem.getInitializer()!=nullptr) elem.getInitializer()->accept(*this);
  // condition
  if (elem.getCondition()!=nullptr) elem.getCondition()->accept(*this);
  // update
  if (elem.getUpdate()!=nullptr) elem.getUpdate()->accept(*this);
  // body
  if (elem.getBody()!=nullptr) elem.getBody()->accept(*this);
  changeToOuterScope();
}

void Visitor::visit(Operator &elem) {}

void Visitor::visit(ParameterList &elem) {
  for (auto &fp : elem.getParameters()) {
    fp->accept(*this);
  }
}

void Visitor::visit(Return &elem) {
  addStatementToScope(elem);
  for (auto &expr : elem.getReturnExpressions()) {
    expr->accept(*this);
  }
}

void Visitor::visit(Rotate &elem) {
  elem.getOperand()->accept(*this);
}

void Visitor::visit(Transpose &elem) {
  elem.getOperand()->accept(*this);
}

void Visitor::visit(UnaryExpr &elem) {
  // operator
  elem.getOperator()->accept(*this);
  // rhs operand
  elem.getRight()->accept(*this);
}

void Visitor::visit(VarAssignm &elem) {
  addStatementToScope(elem);
  elem.getValue()->accept(*this);
}

void Visitor::visit(MatrixAssignm &elem) {
  addStatementToScope(elem);
  elem.getAssignmTarget()->accept(*this);
  elem.getValue()->accept(*this);
}

void Visitor::visit(VarDecl &elem) {
  addStatementToScope(elem);
  // visit datatype associated to new variable
  elem.getDatatype()->accept(*this);
  // visit initializer
  if (elem.getInitializer()!=nullptr) {
    elem.getInitializer()->accept(*this);
  }
}

void Visitor::visit(GetMatrixSize &elem) {
  elem.getMatrixOperand()->accept(*this);
  elem.getDimensionParameter()->accept(*this);
}

void Visitor::visit(Variable &elem) {}

void Visitor::visit(While &elem) {
  addStatementToScope(elem);

  // condition
  elem.getCondition()->accept(*this);

  // then-block
  // if statements following While are nested in a Block, a new scope will be created automatically;
  // if only a single statement is following, we manually need to open a new scope
  if (auto *thenBlock = dynamic_cast<Block *>(elem.getBody())) {
    thenBlock->accept(*this);
  } else {
    auto *block = dynamic_cast<AbstractNode *>(elem.getBody());
    assert(block!=nullptr);
    changeToInnerScope(block->getUniqueNodeId(), &elem);
    elem.getBody()->accept(*this);
    changeToOuterScope();
  }
}

void Visitor::changeToOuterScope() {
  if (ignoreScope) return;
  auto temp = curScope->getOuterScope();
  this->curScope = temp;
}

void Visitor::changeToInnerScope(const std::string &nodeId, AbstractStatement *statement) {
  if (ignoreScope) return;
  auto temp = curScope->getOrCreateInnerScope(nodeId, statement);
  stmtToScopeMapper.insert(std::pair(statement, temp));
  this->curScope = temp;
}

Visitor::Visitor() {
  curScope = nullptr;
}

void Visitor::visit(Datatype &elem) {
  // no children to visit here
}

void Visitor::visit(MatrixElementRef &elem) {
  // operand
  elem.getOperand()->accept(*this);
  // row index
  elem.getRowIndex()->accept(*this);
  // column index
  if (elem.getColumnIndex()!=nullptr) elem.getColumnIndex()->accept(*this);
}

void Visitor::visit(AbstractMatrix &elem) {
  // If this is a Matrix<AbstractExpr*>, we need to call accept(Visitor) on each of its elements.
  if (dynamic_cast<Matrix<AbstractExpr *> * > (&elem)) {
    for (int i = 0; i < elem.getDimensions().numRows; ++i) {
      for (int j = 0; j < elem.getDimensions().numColumns; ++j) {
        // special action required for PrintVisitor only
        if (auto pv = dynamic_cast<PrintVisitor *>(this)) {
          // save the index of this current matrix element in the PrintVisitor (used for string output)
          pv->nextMatrixIndexToBePrinted = std::make_pair(i, j);
        }
        elem.getElementAt(i, j)->accept(*this);
      }
    }
  }
}

void Visitor::visit(OperatorExpr &elem) {
  // visit operator
  elem.getOperator()->accept(*this);
  // visit all operands
  for (auto &child : elem.getOperands()) child->accept(*this);
}

void Visitor::addStatementToScope(AbstractStatement &stat) {
  if (ignoreScope) return;
  if (curScope==nullptr) {
    throw std::logic_error("[Visitor] Cannot add statement to scope as Scope is not created yet (nullptr).");
  }
  curScope->addStatement(&stat);
}

void Visitor::removeStatementFromScope(AbstractStatement &stat) {
  if (ignoreScope) return;
  if (curScope==nullptr) {
    throw std::logic_error("[Visitor] Cannot remove statement from scope as Scope is not created yet (nullptr).");
  }
  curScope->removeStatement(&stat);
}

void Visitor::setIgnoreScope(bool ignScope) {
  Visitor::ignoreScope = ignScope;
}
void Visitor::forceScope(std::unordered_map<AbstractStatement *, Scope *> stmtToScopeMapper, Scope *curScope) {
  this->stmtToScopeMapper = stmtToScopeMapper;
  this->curScope = curScope;
}
