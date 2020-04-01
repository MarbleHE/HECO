#include "Visitor.h"
#include "PrintVisitor.h"
#include "Ast.h"
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
#include "For.h"
#include "LiteralInt.h"
#include "LiteralFloat.h"
#include "LogicalExpr.h"
#include "Operator.h"
#include "Return.h"
#include "Scope.h"
#include "UnaryExpr.h"
#include "VarAssignm.h"
#include "While.h"
#include "Rotate.h"
#include "Transpose.h"
#include "GetMatrixElement.h"
#include "OperatorExpr.h"

void Visitor::visit(Ast &elem) {
  // assumption: AST is always the enclosing object that points to the root
  this->curScope = new Scope("global", nullptr);
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
  changeToInnerScope(elem.getUniqueNodeId());
  for (auto &stat : elem.getStatements()) {
    stat->accept(*this);
  }
  changeToOuterScope();
}

void Visitor::visit(Call &elem) {
  changeToInnerScope(elem.AbstractExpr::getUniqueNodeId());
  // callee
  elem.getFunc()->accept(*this);
  changeToOuterScope();
  // arguments
  for (auto arg : elem.getArguments()) {
    arg->accept(*this);
  }
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
  changeToInnerScope(elem.getUniqueNodeId());
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
  if (auto *thenBranch = dynamic_cast<Block *>(elem.getThenBranch())) {
    thenBranch->accept(*this);
  } else {
    // if thenBranch is no Block we need to manually open a new scope here
    auto thenNode = dynamic_cast<AbstractNode *>(elem.getThenBranch());
    assert(thenNode!=nullptr); // this should never happen
    changeToInnerScope(thenNode->getUniqueNodeId());
    elem.getThenBranch()->accept(*this);
    changeToOuterScope();
  }

  if (elem.getElseBranch()!=nullptr) {
    // elseBranch
    if (auto *elseBranch = dynamic_cast<AbstractNode *>(elem.getElseBranch())) {
      elem.getElseBranch()->accept(*this);
    } else {
      auto elseNode = dynamic_cast<AbstractNode *>(elem.getElseBranch());
      assert(elseNode!=nullptr);
      changeToInnerScope(elseBranch->getUniqueNodeId());
      elem.getElseBranch()->accept(*this);
      changeToOuterScope();
    }
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

  // initializer
  if (elem.getInitializer()!=nullptr) elem.getInitializer()->accept(*this);
  // condition
  if (elem.getCondition()!=nullptr) elem.getCondition()->accept(*this);
  // update
  if (elem.getUpdateStatement()!=nullptr) elem.getUpdateStatement()->accept(*this);

  changeToInnerScope(elem.getStatementToBeExecuted()->getUniqueNodeId());
  // For statement body is always in a separate scope (even without a separate block "{...}")
  elem.getStatementToBeExecuted()->accept(*this);
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

void Visitor::visit(VarDecl &elem) {
  addStatementToScope(elem);
  // visit datatype associated to new variable
  elem.getDatatype()->accept(*this);
  // visit initializer
  if (elem.getInitializer()!=nullptr) {
    elem.getInitializer()->accept(*this);
  }
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
    changeToInnerScope(block->getUniqueNodeId());
    elem.getBody()->accept(*this);
    changeToOuterScope();
  }
}

void Visitor::changeToOuterScope() {
  if (ignoreScope) return;
  auto temp = curScope->getOuterScope();
  this->curScope = temp;
}

void Visitor::changeToInnerScope(const std::string &nodeId) {
  if (ignoreScope) return;
  auto temp = curScope->getOrCreateInnerScope(nodeId);
  this->curScope = temp;
}

Visitor::Visitor() {
  curScope = nullptr;
}

void Visitor::visit(Datatype &elem) {
  // no children to visit here
}

void Visitor::visit(GetMatrixElement &elem) {
  // operand
  elem.getOperand()->accept(*this);
  // row index
  elem.getRowIndex()->accept(*this);
  // column index
  elem.getColumnIndex()->accept(*this);
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

void Visitor::setIgnoreScope(bool ignScope) {
  Visitor::ignoreScope = ignScope;
}
