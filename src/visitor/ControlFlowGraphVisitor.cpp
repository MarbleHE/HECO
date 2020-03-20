#include <GraphNode.h>
#include "ControlFlowGraphVisitor.h"
#include "AbstractNode.h"
#include "ArithmeticExpr.h"
#include "AbstractStatement.h"
#include "Block.h"
#include "For.h"
#include "Function.h"
#include "If.h"
#include "Return.h"
#include "VarAssignm.h"
#include "VarDecl.h"
#include "While.h"
#include "LogicalExpr.h"

void ControlFlowGraphVisitor::visit(Ast &elem) {
  Visitor::visit(elem);
}

// === Statements ====================================
// The Control Flow Graph consists of edges between statements only.

void ControlFlowGraphVisitor::visit(Block &elem) {
  auto gNode = appendStatementToGraph(elem);
  Visitor::visit(elem);
  postActionsStatementVisited(gNode);
}

// Control flow graph for a For statement:
//
//            For statement
//                 │
//                 ▼
//    ┌──── For initializer
//    │
//    │
//    │  For body statement 1 ◀─┐
//    │            │            │
//    │            ▼            │
//    │           ...           │
//    │            │            │
//    │            ▼            │
//    │  For body statement N   │
//    │            │            │
//    │            ▼            │
//    │  For update statement   │
//    │            │            │
//    │            │            │
//    │            ▼            │ condition
//    └───▶ For condition*  ────┘   =True
//                 │
//                 │ condition
//                 │   =False
//                 │
//                 ▼
//        Statement following
//           For statement
//
// (*) Although it's not officially a AbstractStatement in the AST class hierarchy, the condition is treated following
//     as a statement to have it included in the CFG/DFG.
//
void ControlFlowGraphVisitor::visit(For &elem) {
  auto gNode = appendStatementToGraph(elem);

  // initializer (e.g., int i = 0;)
  elem.getInitializer()->accept(*this);
  auto lastStatementInInitializer = lastCreatedNodes;

  // condition (e.g., i <= N)
  handleExpressionsAsStatements = true;
  elem.getCondition()->accept(*this);
  auto lastStatementCondition = lastCreatedNodes;
  handleExpressionsAsStatements = false;

  // body (For (int i = 0; ... ) { body statements })
  elem.getStatementToBeExecuted()->accept(*this);

  // update statement (e.g., i=i+1;)
  elem.getUpdateStatement()->accept(*this);
  auto lastStatementInUpdate = lastCreatedNodes;

  // create an edge from update statement to first statement in for-loop's condition
  auto firstConditionStatement = lastStatementInInitializer.front()->getControlFlowGraph()->getChildren().front();
  for (auto &updateStmt : lastStatementInUpdate) {
    updateStmt->getControlFlowGraph()->addChild(firstConditionStatement);
  }

  // restore the last created nodes in the condition as those need to be connected to the next statement
  lastCreatedNodes = lastStatementCondition;

  postActionsStatementVisited(gNode);
}

void ControlFlowGraphVisitor::visit(Function &elem) {
  auto gNode = appendStatementToGraph(elem);
  Visitor::visit(elem);
  postActionsStatementVisited(gNode);
}

// Control flow graph for an If statement:
//
//                        If statement
//                             │
//                             ▼
//                        If condition*
//                             │
//            condition=True   ▼  condition=False
//            ┌──────────────────────────────────┐
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
// (*) Although it's not officially a AbstractStatement in the AST class hierarchy, the condition is treated following
//     as a statement to have it included in the CFG/DFG.
//
void ControlFlowGraphVisitor::visit(If &elem) {
  auto gNode = appendStatementToGraph(elem);

  // connect the If statement with the If statement's condition
  handleExpressionsAsStatements = true;
  elem.getCondition()->accept(*this);
  handleExpressionsAsStatements = false;
  auto lastStatementCondition = lastCreatedNodes;

  // connect the If condition with the Then-branch
  elem.getThenBranch()->accept(*this);
  auto lastStatementThenBranch = lastCreatedNodes;

  // if existing, visit the Else branch
  std::vector<GraphNode *> lastStatementElseBranch;
  if (elem.getElseBranch()!=nullptr) {
    lastCreatedNodes.clear();
    lastCreatedNodes = lastStatementCondition;
    elem.getElseBranch()->accept(*this);
    lastStatementElseBranch = lastCreatedNodes;
  }

  // add last statement of both branches (then, else)
  lastCreatedNodes.clear();
  lastCreatedNodes.insert(lastCreatedNodes.end(), lastStatementThenBranch.begin(), lastStatementThenBranch.end());
  lastCreatedNodes.insert(lastCreatedNodes.end(), lastStatementElseBranch.begin(), lastStatementElseBranch.end());

  postActionsStatementVisited(gNode);
}

void ControlFlowGraphVisitor::visit(ParameterList &elem) {
  auto gNode = appendStatementToGraph(elem);
  auto defaultAccessModeBackup = defaultAccessMode;
  defaultAccessMode = AccessType::WRITE;
  // visit FunctionParameter
  Visitor::visit(elem);
  defaultAccessMode = defaultAccessModeBackup;
  postActionsStatementVisited(gNode);
}

void ControlFlowGraphVisitor::visit(Return &elem) {
  auto gNode = appendStatementToGraph(elem);
  Visitor::visit(elem);
  postActionsStatementVisited(gNode);
}

void ControlFlowGraphVisitor::visit(VarAssignm &elem) {
  auto gNode = appendStatementToGraph(elem);
  markVariableAccess(elem.getVarTargetIdentifier(), AccessType::WRITE);
  Visitor::visit(elem);
  postActionsStatementVisited(gNode);
}

void ControlFlowGraphVisitor::visit(VarDecl &elem) {
  auto gNode = appendStatementToGraph(elem);
  markVariableAccess(elem.getVarTargetIdentifier(), AccessType::WRITE);
  Visitor::visit(elem);
  postActionsStatementVisited(gNode);
}

// Control flow graph for a While statement:
//
//         While Statement
//                │
//                ▼
//    ┌──  While Condition*  ◀─┐
//    │           │            │
//    │           ▼            │
//    │      While Body        │
//    │     Statement 1        │
//    │           │            │
//    │           │            │
//    │           ▼            │
//    │          ...           │
//    │           │            │
//    │           │            │
//    │           ▼            │
//    │      While Body        │
//    │     Statement N   ─────┘
//    │
//    └───▶ Next Statement
//
// (*) Although it's not officially a AbstractStatement in the AST class hierarchy, the condition is treated following
//     as a statement to have it included in the CFG/DFG.
//
void ControlFlowGraphVisitor::visit(While &elem) {
  auto gNode = appendStatementToGraph(elem);

  handleExpressionsAsStatements = true;
  elem.getCondition()->accept(*this);
  handleExpressionsAsStatements = false;
  auto conditionStmt = lastCreatedNodes;

  elem.getBody()->accept(*this);

  // add this While statement as child for each node in lastCreatedNodes
  for (auto &c : lastCreatedNodes) {
    c->getControlFlowGraph()->addChild(conditionStmt.back());
  }

  lastCreatedNodes = conditionStmt;
  postActionsStatementVisited(gNode);
}

// === Expressions ===================================

void ControlFlowGraphVisitor::visit(ArithmeticExpr &elem) {
  if (handleExpressionsAsStatements) {
    auto gNode = appendStatementToGraph(elem);
    Visitor::visit(elem);
    postActionsStatementVisited(gNode);
  } else {
    Visitor::visit(elem);
  }
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
  if (handleExpressionsAsStatements) {
    auto gNode = appendStatementToGraph(elem);
    handleExpressionsAsStatements = false;
    Visitor::visit(elem);
    postActionsStatementVisited(gNode);
  } else {
    Visitor::visit(elem);
  }
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
  markVariableAccess({elem});
  Visitor::visit(elem);
}

void ControlFlowGraphVisitor::visit(GetMatrixElement &elem) {
  Visitor::visit(elem);
}

// ===================================================

GraphNode *ControlFlowGraphVisitor::appendStatementToCfg(AbstractNode &abstractStmt,
                                                         const std::vector<GraphNode *> &parentNodes) {
  auto node = new GraphNode(&abstractStmt);
  if (parentNodes.empty()) {
    rootNode = node;
  }
  for (auto &p : parentNodes) {
    p->getControlFlowGraph()->addChild(node);
  }

  lastCreatedNodes.clear();
  lastCreatedNodes.push_back(node);

  return node;
}

GraphNode *ControlFlowGraphVisitor::appendStatementToGraph(AbstractNode &abstractStmt) {
  return appendStatementToCfg(abstractStmt, lastCreatedNodes);
}

GraphNode *ControlFlowGraphVisitor::getRootNode() const {
  return rootNode;
}

void ControlFlowGraphVisitor::postActionsStatementVisited(GraphNode *gNode) {
  if (gNode!=nullptr) {
    gNode->setAccessedVariables(std::move(varAccess));
  } else {
    varAccess.clear();
  }
}

void ControlFlowGraphVisitor::markVariableAccess(const std::string &variableIdentifier, AccessType accessMode) {
  varAccess.insert(std::make_pair(variableIdentifier, accessMode));
}

void ControlFlowGraphVisitor::markVariableAccess(Variable &var) {
  markVariableAccess(var.getIdentifier(), defaultAccessMode);
}
