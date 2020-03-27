#include "ControlFlowGraphVisitor.h"
#include <queue>
#include <tuple>
#include "GraphNode.h"
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
#include "OperatorExpr.h"

void ControlFlowGraphVisitor::visit(Ast &elem) {
  Visitor::visit(elem);
}

// === Statements ====================================
// The Control Flow Graph consists of edges between statements only.

void ControlFlowGraphVisitor::visit(Block &elem) {
  auto gNode = appendStatementToCfg(elem);
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
  auto gNode = appendStatementToCfg(elem);

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
  auto gNode = appendStatementToCfg(elem);
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
  auto gNode = appendStatementToCfg(elem);

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

    // add last statement of both branches (then, else)
    lastCreatedNodes.clear();
    lastCreatedNodes.insert(lastCreatedNodes.end(), lastStatementThenBranch.begin(), lastStatementThenBranch.end());
    lastCreatedNodes.insert(lastCreatedNodes.end(), lastStatementElseBranch.begin(), lastStatementElseBranch.end());
  } else {
    // add the if statement's condition (if condition is False) and the last statement in the if statement's body
    lastCreatedNodes.clear();
    lastCreatedNodes.insert(lastCreatedNodes.end(), lastStatementCondition.begin(), lastStatementCondition.end());
    lastCreatedNodes.insert(lastCreatedNodes.end(), lastStatementThenBranch.begin(), lastStatementThenBranch.end());
  }
  postActionsStatementVisited(gNode);
}

void ControlFlowGraphVisitor::visit(ParameterList &elem) {
  auto gNode = appendStatementToCfg(elem);
  auto defaultAccessModeBackup = defaultAccessMode;
  defaultAccessMode = AccessType::WRITE;
  // visit FunctionParameter
  Visitor::visit(elem);
  defaultAccessMode = defaultAccessModeBackup;
  postActionsStatementVisited(gNode);
}

void ControlFlowGraphVisitor::visit(Return &elem) {
  auto gNode = appendStatementToCfg(elem);
  Visitor::visit(elem);
  postActionsStatementVisited(gNode);
}

void ControlFlowGraphVisitor::visit(VarAssignm &elem) {
  auto gNode = appendStatementToCfg(elem);
  markVariableAccess(elem.getVarTargetIdentifier(), AccessType::WRITE);
  Visitor::visit(elem);
  postActionsStatementVisited(gNode);
}

void ControlFlowGraphVisitor::visit(VarDecl &elem) {
  auto gNode = appendStatementToCfg(elem);
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
  auto gNode = appendStatementToCfg(elem);

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

void ControlFlowGraphVisitor::handleOperatorExpr(AbstractExpr &ae) {
  if (handleExpressionsAsStatements) {
    auto gNode = appendStatementToCfg(ae);
    handleExpressionsAsStatements = false;
    Visitor::visit(ae);
    postActionsStatementVisited(gNode);
  } else {
    Visitor::visit(ae);
  }
}

void ControlFlowGraphVisitor::visit(ArithmeticExpr &elem) {
  handleOperatorExpr(elem);
}

void ControlFlowGraphVisitor::visit(LogicalExpr &elem) {
  handleOperatorExpr(elem);
}

void ControlFlowGraphVisitor::visit(OperatorExpr &elem) {
  handleOperatorExpr(elem);
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

GraphNode *ControlFlowGraphVisitor::appendStatementToCfg(AbstractNode &node) {
  return appendStatementToCfg(node, lastCreatedNodes);
}

GraphNode *ControlFlowGraphVisitor::getRootNodeCfg() const {
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

void ControlFlowGraphVisitor::buildDataFlowGraph() {
  // =================
  // STEP 1:
  // Traverse the graph and store for each graph node where (i.e., at which node) all of the variables seen so far were
  // written last time.
  // =================

  // a temporary map to remember the statements where a variable was last written, this is used as temporary storage
  // for the node currently processed (curNode)
  std::unordered_map<std::string, std::unordered_set<GraphNode *>> varLastWritten;

  // a map to remember for each GraphNode where all the variables visited on the way to the node were last written
  std::unordered_map<GraphNode *, std::unordered_map<std::string, std::unordered_set<GraphNode *>>>
      nodeToVarLastWrittenMapping;

  // a set to recognize already visited nodes, this is needed because our CFG potentially contains cycles
  std::set<GraphNode *> processedNodes;

  // iterate forwards through CFG in BFS and remember the GraphNode where a variable was last written
  std::queue<GraphNode *> nextNodeToVisit({getRootNodeCfg()});
  while (!nextNodeToVisit.empty()) {
    auto curNode = nextNodeToVisit.front();
    nextNodeToVisit.pop();

    // merge the information about last writes from all curNode parents (incoming edges) temporarily into varLastWritten
    varLastWritten.clear();
    auto parentNodesVec = curNode->getControlFlowGraph()->getParents();
    bool nodeIsJoinPoint = parentNodesVec.size() > 1;  // joint point: CFG node where multiple nodes flow together
    // for each parent node
    for (auto &pNode : parentNodesVec) {
      // skip this parent if it was not visited before -> no information to collect available
      if (nodeToVarLastWrittenMapping.count(pNode)==0) continue;
      // go though all variables for which the parent has registered writes
      for (auto &[varIdentifier, vectorOfReferencedNodes] : nodeToVarLastWrittenMapping.at(pNode)) {
        // either add the nodes that refer to the variable (then) in case that this variable is already known [merging],
        // or (else) create a new vector using the vector of referenced nodes [replacing]
        if (varLastWritten.count(varIdentifier) > 0) {
          for (auto &val : vectorOfReferencedNodes) varLastWritten.at(varIdentifier).insert(val);
        } else {
          varLastWritten[varIdentifier] = vectorOfReferencedNodes;
        }
      }
    }

    // add writes to variables happening in curNode to varLastWritten
    for (auto &[varIdentifier, ignored] : curNode->getVariables(AccessType::WRITE)) {
      // store the variable writes of this node (curNode)
      if (varLastWritten.count(varIdentifier) > 0) {
        // if this is not a join point, we need to remove the existing information before adding the new one
        if (!nodeIsJoinPoint) varLastWritten.at(varIdentifier).clear();
        varLastWritten.at(varIdentifier).insert(curNode);
      } else {
        varLastWritten[varIdentifier] = std::unordered_set<GraphNode *>({curNode});
      }
    }

    // compare varLastWritten with information in nodeToVarLastWrittenMapping to see whether there were any changes such
    // that the newly collected knowledge needs to be distributed to the children nodes again -> revisit required
    // (check will be evaluated only if the node was visited once before, see condition in visitingChildrenRequired)
    auto collectedWrittenVarsChanged = [&]() -> bool {
      // compare for each identifier in varLastWritten if the set of collected nodes is different from the set of
      // already collected nodes (see nodeToVarLastWrittenMapping)
      auto mp = nodeToVarLastWrittenMapping.at(curNode);
      for (auto &[varIdentifier, gNodeSet] : varLastWritten) {
        if (mp.count(varIdentifier)==0 || (gNodeSet!=mp.at(varIdentifier))) { return true; }
      }
      return false;
    };
    // condition that decides whether children must be enqueued / visited next
    bool visitingChildrenRequired = (processedNodes.count(curNode)==0)  // node was not visited yet
        || (nodeToVarLastWrittenMapping.count(curNode) > 0 && collectedWrittenVarsChanged()); // information changed

    // attach the collected write information to this node, for that it is required to check if there is already
    // existing information in nodeToVarLastWrittenMapping about this node (then append) or not (then move/overwrite)
    if (nodeToVarLastWrittenMapping.count(curNode)==0) {
      // simply move the collected nodes in varLastWritten to nodeToVarLastWrittenMapping
      nodeToVarLastWrittenMapping[curNode] = std::move(varLastWritten);
    } else {
      // merge the nodes already existing in nodeToVarLastWrittenMapping with those newly collected
      for (auto &[varIdentifier, gNodeSet] : varLastWritten) {
        auto set = varLastWritten.at(varIdentifier);
        nodeToVarLastWrittenMapping.at(curNode).at(varIdentifier).insert(set.begin(), set.end());
      }
    }

    // stop loop iteration here, if (re)visiting the child nodes is not required
    if (!visitingChildrenRequired) continue;

    // enqueue children of current node
    for (auto &it : curNode->getControlFlowGraph()->getChildren()) nextNodeToVisit.push(it);
    processedNodes.insert(curNode);
  }

  // =================
  // STEP 2:
  // Traverse all graph nodes that have variable reads and add an edge to the last location where the respective
  // variable was written last.
  // =================

  // for each node that was visited in the CFG
  for (auto &v : processedNodes) {
    // retrieve all variables that were read
    for (auto &[varIdentifierRead, ignored1] : v->getVariables(AccessType::READ)) {
      // SPECIAL CASE: node has a WRITE to the same variable (=> READ + WRITE, e.g., i = i + 1), in that case it does
      // not make sense to add a self-edge, but in case that the node is within a loop, its parent node will have the
      // same information about the last write
      if (v->getVariables(AccessType::WRITE).count(std::make_pair(varIdentifierRead, AccessType::WRITE)) > 0) {
        // iterate over all parents of node v
        for (auto &parentNode : v->getControlFlowGraph()->getParents()) {
          // if the parent knows where the last write for the given variable identifier happened lastly
          if (nodeToVarLastWrittenMapping.at(parentNode).count(varIdentifierRead)==0) continue;
          // then create an edge from each of the nodes that have written to the variable recently to this node v
          for (auto &edgeSrc : nodeToVarLastWrittenMapping.at(parentNode).at(varIdentifierRead))
            edgeSrc->getDataFlowGraph()->addChild(v);
        }
      } else { // DEFAULT CASE
        // add an bilateral edge (last node that wrote to variable, current node) to each of the variables that last
        // wrote to the variable (e.g., in case of a branch statement, it can be multiple nodes)
        for (auto &writeNodes : nodeToVarLastWrittenMapping.at(v).at(varIdentifierRead))
          writeNodes->getDataFlowGraph()->addChild(v);
      }
    }
  }
}
