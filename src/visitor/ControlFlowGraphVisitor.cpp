#include <queue>
#include <tuple>
#include <algorithm>
#include "ast_opt/visitor/ControlFlowGraphVisitor.h"
#include "ast_opt/utilities/GraphNode.h"
#include "ast_opt/ast/AbstractNode.h"
#include "ast_opt/ast/ArithmeticExpr.h"
#include "ast_opt/ast/Block.h"
#include "ast_opt/ast/For.h"
#include "ast_opt/ast/Function.h"
#include "ast_opt/ast/If.h"
#include "ast_opt/ast/Return.h"
#include "ast_opt/ast/VarAssignm.h"
#include "ast_opt/ast/VarDecl.h"
#include "ast_opt/ast/While.h"
#include "ast_opt/ast/LogicalExpr.h"
#include "ast_opt/ast/OperatorExpr.h"
#include "ast_opt/ast/MatrixAssignm.h"

void ControlFlowGraphVisitor::visit(Ast &elem) {
  // reset all custom variables
  variablesReadAndWritten.clear();
  lastCreatedNodes.clear();
  varAccess.clear();
  rootNode = nullptr;
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

  // Manual handling of scope (usually done via Visitor::visit(elem))
  addStatementToScope(elem);
  changeToInnerScope(elem.getUniqueNodeId(), &elem);

  // initializer (e.g., int i = 0;)
  // Manually visit the statements in the block, since otherwise Visitor::visit would create a new scope!
  if (elem.getInitializer()!=nullptr) {
    for (auto &s: elem.getInitializer()->getStatements()) {
      s->accept(*this);
    }
  }
  auto lastStatementInInitializer = lastCreatedNodes;

  // condition (e.g., i <= N)
  handleExpressionsAsStatements = true;
  elem.getCondition()->accept(*this);
  auto lastStatementCondition = lastCreatedNodes;
  handleExpressionsAsStatements = false;

  // body (For (int i = 0; ... ) { body statements })
  elem.getBody()->accept(*this);
  auto lastStatementInBody = lastCreatedNodes;

  // update statement (e.g., i=i+1;)
  std::vector<GraphNode*> lastStatementInUpdate;
  if (elem.getUpdate()!=nullptr) {
    elem.getUpdate()->accept(*this);
    lastStatementInUpdate = lastCreatedNodes;
  }

  // TODO if there is an update statement and a condition

  // TODO otherwise ...

  // create an edge from update statement to first statement in for-loop's condition
  auto firstConditionStatement = lastStatementInInitializer.front()->getControlFlowGraph()->getChildren().front();
  for (auto &updateStmt : lastStatementInUpdate) {
    updateStmt->getControlFlowGraph()->addChild(firstConditionStatement);
  }

  // restore the last created nodes in the condition as those need to be connected to the next statement
  lastCreatedNodes = lastStatementCondition;

  // Manual scope handling
  changeToOuterScope();

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
// (*) Although it is not officially an AbstractStatement in the AST class hierarchy, the condition is treated following
//     like a statement to have it included in the CFG/DFG.
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
  auto scopedVar = variableValues.getVariableEntryDeclaredInThisOrOuterScope(elem.getVarTargetIdentifier(), curScope);
  markVariableAccess(scopedVar, AccessType::WRITE);
  Visitor::visit(elem);
  postActionsStatementVisited(gNode);
}

void ControlFlowGraphVisitor::visit(MatrixAssignm &elem) {
  auto gNode = appendStatementToCfg(elem);
  // TODO Make the varAccess structure more flexible to allow storing MatrixElementRef and Variable objects, instead
  //  of std::string objects only. Also consider extending varAccess' key to use a (std::string, Scope*) pair to
  //  uniquely identify a variable.
  // This temporary workaround uses a string representation of the assignment target, for example,
  //    Variable (M) [LiteralInt (32)][LiteralInt (1)]
  // to refer to the element at (32,1) of matrix M. This does not work well because there might exist different index
  // expressions pointing to the same element (e.g., M[a][b] == M[b][d] if a==b and b==d), hence we cannot easily
  // distinguish matrix accesses.
  // TODO: Currently, we simply consider the entire matrix variable, not the individual indices
  auto scopedVar = variableValues.getVariableEntryDeclaredInThisOrOuterScope(getVarTargetIdentifier(&elem), curScope);
  markVariableAccess(scopedVar, AccessType::WRITE);
  Visitor::visit(elem);
  postActionsStatementVisited(gNode);
}

void ControlFlowGraphVisitor::visit(VarDecl &elem) {
  // Visit and simplify datatype and initializer (if present)
  Visitor::visit(elem);

  // store the variable, but ignore value since we don't care about that in CFGV
  auto sv = ScopedVariable(elem.getIdentifier(), curScope);
  variableValues.addDeclaredVariable(sv, VariableValue(*elem.getDatatype(), nullptr));

  auto gNode = appendStatementToCfg(elem);
  markVariableAccess(sv, AccessType::WRITE);
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
// (*) Although it is not officially an AbstractStatement in the AST class hierarchy, the condition is treated following
//     like a statement to have it included in the CFG/DFG.
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
  // We cannot simply visit the Variable, as it would try to look it up when of course it does not exist yet
  // So instead of Visitor::visit(elem); we visit only the Datatype and inspect the Value manually
  elem.getDatatype()->accept(*this);

  // The value in a FunctionParamter must be a single Variable
  auto valueAsVar = dynamic_cast<Variable *>(elem.getValue());
  if (valueAsVar) {
    // Ignoring value, as it's irrelevant for CFGV
    variableValues.addDeclaredVariable(ScopedVariable(valueAsVar->getIdentifier(), curScope), VariableValue(*elem.getDatatype(), nullptr));
  } else {
    throw std::runtime_error("Function parameter " + elem.getUniqueNodeId() + " contained invalid Variable value.");
  }
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
  auto sv = variableValues.getVariableEntryDeclaredInThisOrOuterScope(elem.getIdentifier(), curScope);
  markVariableAccess(sv);
  Visitor::visit(elem);
}

void ControlFlowGraphVisitor::visit(MatrixElementRef &elem) {
  Visitor::visit(elem);
}

void ControlFlowGraphVisitor::visit(GetMatrixSize &elem) {
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

void ControlFlowGraphVisitor::markVariableAccess(const ScopedVariable &var) {
  markVariableAccess(var, defaultAccessMode);
}

void ControlFlowGraphVisitor::markVariableAccess(const ScopedVariable &var, AccessType accessType) {
  varAccess.insert(std::make_pair(var, accessType));
}

void ControlFlowGraphVisitor::buildDataFlowGraph() {
  // =================
  // STEP 1:
  // Traverse the graph and store for each graph node where (i.e., at which node) all of the variables seen so far were
  // written last time.
  // =================

  // a temporary map to remember the statements where a variable was last written, this is used as temporary storage
  // for the node currently processed (curNode)
  std::map<ScopedVariable, std::unordered_set<GraphNode *>> varLastWritten;

  // a map to remember for each GraphNode where all the variables visited on the way to the node were last written
  std::map<GraphNode *, std::map<ScopedVariable, std::unordered_set<GraphNode *>>>
      nodeToVarLastWrittenMapping;

  //TODO: Can we maybe ignore the entire variableValues thing in the main CFG visitor
  // and instead just pass in VariableValues into this function?

  // TODO: Anything stored in VariableValues should be associated with having been "last written" in the root node?
  std::unordered_set us = {getRootNodeCfg()}; //necessary to disambiguate calls
  for (auto &[sv, vv]: variableValues.getMap()) {
    varLastWritten.insert_or_assign(sv, us);
  }
  nodeToVarLastWrittenMapping.insert_or_assign(getRootNodeCfg(), varLastWritten);
  varLastWritten.clear();

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
      for (auto &[var, vectorOfReferencedNodes] : nodeToVarLastWrittenMapping.at(pNode)) {
        // either add the nodes that refer to the variable (then) in case that this variable is already known [merging],
        // or (else) create a new vector using the vector of referenced nodes [replacing]
        if (varLastWritten.count(var) > 0) {
          for (auto &val : vectorOfReferencedNodes) varLastWritten.at(var).insert(val);
        } else {
          varLastWritten[var] = vectorOfReferencedNodes;
        }
      }
    }

    // add writes to variables happening in curNode to varLastWritten
    for (auto &var : curNode->getVariables(AccessType::WRITE)) {
      // store the variable writes of this node (curNode)
      if (varLastWritten.count(var) > 0) {
        // if this is not a join point, we need to remove the existing information before adding the new one
        if (!nodeIsJoinPoint) varLastWritten.at(var).clear();
        varLastWritten.at(var).insert(curNode);
      } else {
        varLastWritten[var] = std::unordered_set<GraphNode *>({curNode});
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
        //check if the variable already exists at this node, if not, create it
        if (nodeToVarLastWrittenMapping.at(curNode).find(varIdentifier)
            ==nodeToVarLastWrittenMapping.at(curNode).end()) {
          nodeToVarLastWrittenMapping.at(curNode).insert(std::make_pair<>(varIdentifier, set));
        } else {
          nodeToVarLastWrittenMapping.at(curNode).at(varIdentifier).insert(set.begin(), set.end());
        }
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
    for (auto &var : v->getVariables(AccessType::READ)) {
      // SPECIAL CASE: node has a WRITE to the same variable (=> READ + WRITE, e.g., i = i + 1), in that case it does
      // not make sense to add a self-edge, but in case that the node is within a loop, its parent node will have the
      // same information about the last write
      if (v->getVariables(AccessType::WRITE).count(var) > 0) {
        // iterate over all parents of node v
        for (auto &parentNode : v->getControlFlowGraph()->getParents()) {
          // if the parent knows where the last write for the given variable identifier happened lastly
          if (nodeToVarLastWrittenMapping.at(parentNode).count(var)==0) continue;
          // then create an edge from each of the nodes that have written to the variable recently to this node v
          for (auto &edgeSrc : nodeToVarLastWrittenMapping.at(parentNode).at(var))
            edgeSrc->getDataFlowGraph()->addChild(v);
        }
      } else { // DEFAULT CASE
        // add an bilateral edge (last node that wrote to variable, current node) to each of the variables that last
        // wrote to the variable (e.g., in case of a branch statement, it can be multiple nodes)
        if (nodeToVarLastWrittenMapping.find(v)==nodeToVarLastWrittenMapping.end()) {
          throw std::logic_error(
              "CFGV expected node corresponding to " + v->getRefToOriginalNode()->getUniqueNodeId()
                  + " to have a LastWrittenMap but it did not.");
        } else if (nodeToVarLastWrittenMapping.at(v).find(var)==nodeToVarLastWrittenMapping.at(v).end()) {
          throw std::runtime_error("CFGV: Found uninitialized (i.e. never written) variable: "
                                       + var.getIdentifier() + " in statement "
                                       + v->getRefToOriginalNode()->getUniqueNodeId());
        } else {
          for (auto &writeNodes : nodeToVarLastWrittenMapping.at(v).at(var))
            writeNodes->getDataFlowGraph()->addChild(v);
        }
      }
    }
  }

  // =================
  // STEP 3:
  // Traverse all graph nodes and collect all variable reads and writes to build variablesReadAndWritten
  // =================
  std::vector<ScopedVariable> written;
  std::vector<ScopedVariable> read;
  for (auto &n : processedNodes) {
    auto w = n->getVariables(AccessType::WRITE);
    std::copy(w.begin(), w.end(), std::back_inserter(written));
    auto r = n->getVariables(AccessType::READ);
    std::copy(r.begin(), r.end(), std::back_inserter(read));
  }
  // determine the variables that were read and written (must be both!)
  std::sort(written.begin(), written.end());
  std::sort(read.begin(), read.end());
  // Intersection requires sorted inputs
  std::set_intersection(written.begin(),
                        written.end(),
                        read.begin(),
                        read.end(),
                        std::inserter(variablesReadAndWritten, variablesReadAndWritten.begin()));
}

std::set<ScopedVariable> ControlFlowGraphVisitor::getVariablesReadAndWritten() {
  return variablesReadAndWritten;
}

void ControlFlowGraphVisitor::forceVariableValues(const VariableValuesMap &variableValues) {
  this->variableValues = variableValues;
}