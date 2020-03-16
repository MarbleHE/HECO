#include "ControlFlowGraphVisitor.h"
#include "AbstractNode.h"
#include "AbstractStatement.h"
#include "Block.h"
#include "For.h"
#include "Function.h"
#include "If.h"
#include "Return.h"
#include "VarAssignm.h"
#include "VarDecl.h"
#include "While.h"

void ControlFlowGraphVisitor::visit(Ast &elem) {
  Visitor::visit(elem);
}

// === Statements ====================================
// The Control Flow Graph consists of edges between statements only.

void ControlFlowGraphVisitor::visit(Block &elem) {
  appendStatementToGraph(elem);
  Visitor::visit(elem);
}

// Control flow graph for a For statement:
//
//          For statement
//               │
//               ▼
//  ┌──── For initializer
//  │
//  │
//  │  For body statement 1 ◀─┐
//  │            │            │
//  │            ▼            │
//  │           ...           │
//  │            │            │
//  │            ▼            │
//  │  For body statement N   │
//  │            │            │
//  │            ▼            │
//  │  For update statement   │
//  │            │            │
//  │            │            │
//  │            ▼            │
//  │    [check condition]    │ condition
//  └─▶   (no statement)    ──┘   =True
//               │
//               │ condition
//               │   =False
//               │
//               ▼
//      Statement following
//         For statement
//
// Note that the For condition is not a statement but an expression, therefore this is not a node in the CFG and is
// passed over instead. For example, the For initializer is connected directly with the first statement in the For
// statement's body.
void ControlFlowGraphVisitor::visit(For &elem) {
  appendStatementToGraph(elem);

  // initializer (e.g., int i = 0;)
  elem.getInitializer()->accept(*this);
  auto lastStatementInInitializer = lastCreatedNodes;

  // body (For (int i = 0; ... ) { body statements})
  elem.getStatementToBeExecuted()->accept(*this);

  // update statement (e.g., i=i+1;)
  elem.getUpdateStatement()->accept(*this);
  auto lastStatementInUpdate = lastCreatedNodes;

  // create an edge from update statement to first statement in for-loop's body
  auto firstBodyStatement = lastStatementInInitializer.front()->children.front();
  for (auto &updateStmt : lastStatementInUpdate) {
    updateStmt->addChild(firstBodyStatement);
  }

  // Update the lastCreatedNodes as the next statement must be attached to the For statement node as well as to the
  // update statement. If the For-loop is not executed once, then the next statement to be executed is the one following
  // the For statement (after the For statement's body), otherwise the For loop is executed as long as the condition is
  // True such we need an edge from the update statement to the statement after the For statement.
  lastCreatedNodes.clear();
  lastCreatedNodes.insert(lastCreatedNodes.end(), lastStatementInInitializer.begin(), lastStatementInInitializer.end());
  lastCreatedNodes.insert(lastCreatedNodes.end(), lastStatementInUpdate.begin(), lastStatementInUpdate.end());
}

void ControlFlowGraphVisitor::visit(Function &elem) {
  appendStatementToGraph(elem);
  Visitor::visit(elem);
}

// Control flow graph for an If statement:
//
//                        If statement
//                             │
//             condition=True  │ condition=False
//            ┌────────────────┴─────────────────┐
//            │                                  │
//            ▼                                  ▼
//      Then branch                        Else branch
//      statement 1                        statement 1
//            │                                  │
//            ▼                                  ▼
//           ...                                ...
//            │                                  │
//            ▼                                  ▼
//      Then branch                        Else branch
//      statement N                        statement N
//            │                                  │
//            └─────▶   Next statement    ◀──────┘
//
void ControlFlowGraphVisitor::visit(If &elem) {
  auto ifStmtNode = appendStatementToGraph(elem);

  // connect the If statement with the first statement in the Then branch
  elem.getThenBranch()->accept(*this);
  auto lastStatementThenBranch = lastCreatedNodes;
  lastCreatedNodes.clear();
  lastCreatedNodes.push_back(ifStmtNode);

  // if existing, visit the Else branch
  std::vector<GraphNode *> lastStatementElseBranch;
  if (elem.getElseBranch()!=nullptr) {
    elem.getElseBranch()->accept(*this);
    lastStatementElseBranch = lastCreatedNodes;
  }

  // add last statement of both branches (then, else)
  lastCreatedNodes.clear();
  lastCreatedNodes.insert(lastCreatedNodes.end(), lastStatementThenBranch.begin(), lastStatementThenBranch.end());
  lastCreatedNodes.insert(lastCreatedNodes.end(), lastStatementElseBranch.begin(), lastStatementElseBranch.end());
}

void ControlFlowGraphVisitor::visit(ParameterList &elem) {
  Visitor::visit(elem);
}

void ControlFlowGraphVisitor::visit(Return &elem) {
  appendStatementToGraph(elem);
  Visitor::visit(elem);
}

void ControlFlowGraphVisitor::visit(VarAssignm &elem) {
  appendStatementToGraph(elem);
  Visitor::visit(elem);
}

void ControlFlowGraphVisitor::visit(VarDecl &elem) {
  appendStatementToGraph(elem);
  Visitor::visit(elem);
}

void ControlFlowGraphVisitor::visit(While &elem) {
  auto whileGraphNode = appendStatementToGraph(elem);
  auto lastStatementWhile = lastCreatedNodes;
  elem.getBody()->accept(*this);

  // add this While statement as child for each node in lastCreatedNodes
  for (auto &c : lastCreatedNodes) {
    c->addChild(whileGraphNode);
  }

  // restore lastCreatedNodes such that statement following While is connected with While statement instead of the last
  // statement in the While's body
  lastCreatedNodes = lastStatementWhile;
}

// === Expressions ===================================

void ControlFlowGraphVisitor::visit(ArithmeticExpr &elem) {
  Visitor::visit(elem);
}

void ControlFlowGraphVisitor::visit(Call &elem) {
  Visitor::visit(elem);
}

void ControlFlowGraphVisitor::visit(CallExternal &elem) {
  Visitor::visit(elem);
}

void ControlFlowGraphVisitor::visit(Datatype &elem) {
  Visitor::visit(elem);
}

void ControlFlowGraphVisitor::visit(FunctionParameter &elem) {
  Visitor::visit(elem);
}

void ControlFlowGraphVisitor::visit(LiteralBool &elem) {
  Visitor::visit(elem);
}

void ControlFlowGraphVisitor::visit(LiteralInt &elem) {
  Visitor::visit(elem);
}

void ControlFlowGraphVisitor::visit(LiteralString &elem) {
  Visitor::visit(elem);
}

void ControlFlowGraphVisitor::visit(LiteralFloat &elem) {
  Visitor::visit(elem);
}

void ControlFlowGraphVisitor::visit(LogicalExpr &elem) {
  Visitor::visit(elem);
}

void ControlFlowGraphVisitor::visit(Operator &elem) {
  Visitor::visit(elem);
}

void ControlFlowGraphVisitor::visit(Rotate &elem) {
  Visitor::visit(elem);
}

void ControlFlowGraphVisitor::visit(Transpose &elem) {
  Visitor::visit(elem);
}

void ControlFlowGraphVisitor::visit(UnaryExpr &elem) {
  Visitor::visit(elem);
}

void ControlFlowGraphVisitor::visit(Variable &elem) {
  Visitor::visit(elem);
}

void ControlFlowGraphVisitor::visit(GetMatrixElement &elem) {
  Visitor::visit(elem);
}

// ===================================================

GraphNode *ControlFlowGraphVisitor::appendStatementToGraph(AbstractStatement &abstractStmt,
                                                           std::vector<GraphNode *> parentNodes) {
  auto node = new GraphNode(abstractStmt.castTo<AbstractNode>());
  if (parentNodes.empty()) {
    rootNode = node;
  }
  for (auto &p : parentNodes) {
    p->addChild(node);
  }

  lastCreatedNodes.clear();
  lastCreatedNodes.push_back(node);

  return node;
}

GraphNode *ControlFlowGraphVisitor::appendStatementToGraph(AbstractStatement &abstractStmt) {
  return appendStatementToGraph(abstractStmt, lastCreatedNodes);
}

GraphNode *ControlFlowGraphVisitor::getRootNode() const {
  return rootNode;
}



