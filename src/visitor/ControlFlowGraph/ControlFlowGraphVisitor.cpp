#include "ast_opt/visitor/ControlFlowGraph/ControlFlowGraphVisitor.h"
#include "ast_opt/ast/IndexAccess.h"
#include "ast_opt/utilities/Scope.h"
#include "ast_opt/ast/Assignment.h"
#include "ast_opt/ast/Block.h"
#include "ast_opt/ast/For.h"
#include "ast_opt/ast/Function.h"
#include "ast_opt/ast/If.h"
#include "ast_opt/ast/Return.h"
#include "ast_opt/ast/VariableDeclaration.h"

void SpecialControlFlowGraphVisitor::checkEntrypoint(AbstractNode &node) {
  // if this is not the first visited node, then there's no need for this check
  if (!nodes.empty()) return;

  // if this is the first visited node ---
  // make sure that CFGV was called on a Block, If, or For node
  auto nodeAsBlock = dynamic_cast<Block *>(&node);
  auto nodeAsIf = dynamic_cast<If *>(&node);
  auto nodeAsFor = dynamic_cast<For *>(&node);

  if (nodeAsBlock==nullptr && nodeAsIf==nullptr && nodeAsFor==nullptr) {
    throw std::runtime_error(
        "Cannot run ControlFlowGraphVisitor on given node (" + node.getUniqueNodeId() + "). " +
            "Visitor requires a Block, For, or If AST node as (root) input.");
  }
}

void SpecialControlFlowGraphVisitor::visit(Assignment &node) {
  SpecialControlFlowGraphVisitor::checkEntrypoint(node);
  std::cout << "Visiting Assignment (" << node.getUniqueNodeId() << ")" << std::endl;
  GraphNode &graphNode = createGraphNodeAndAppendToCfg(node);

  std::string identifier;
  if (auto variable = dynamic_cast<Variable *>(&node.getTarget())) {
    identifier = variable->getIdentifier();
  } else if (auto indexAccess = dynamic_cast<IndexAccess *>(&node.getTarget())) {
    // TODO implement me: must recursively go tree down and retrieve all variable identifiers
  }

  // visit the right-hand side of the assignment
  node.getValue().accept(*this);

  if (getCurrentScope().identifierExists(identifier) || !ignoreNonDeclaredVariables) {
    markVariableAccess(getCurrentScope().resolveIdentifier(identifier), VariableAccessType::WRITE);
  }

  storeAccessedVariables(graphNode);
}

void SpecialControlFlowGraphVisitor::visit(Block &node) {
  SpecialControlFlowGraphVisitor::checkEntrypoint(node);
  std::cout << "Visiting Block (" << node.getUniqueNodeId() << ")" << std::endl;
  GraphNode &graphNode = createGraphNodeAndAppendToCfg(node);
  ScopedVisitor::visit(node);
  storeAccessedVariables(graphNode);
}

////////////////////////////////////
///////////// For Statement ////////
////////////////////////////////////
// CFG Graph
//
//        Initializer Stmt. 1
//                 │
//                 ▼
//    ┌── Initializer Stmt. N
//    │
//    │
//    │      Body Stmt. 1    ◀─┐
//    │            │           │
//    │            ▼           │
//    │      Body Stmt. N      │
//    │            │           │
//    │            │           │
//    │            │           │
//    │            ▼           │
//    │     Update Stmt. 1     │
//    │            │           │
//    │            ▼           │
//    │     Update Stmt. N     │
//    │            │           │
//    │            │           │ condition
//    │            ▼           │  == true
//    └─────▶ Condition*  ─────┘
//                 │
//                 │   condition
//                 │   == false
//                 ▼
//            Next Stmt.
//
// (*)Note: Although the Condition is not a AbstractStatement in our AST model, we include it here as we need the
// information which variables are accessed.
void SpecialControlFlowGraphVisitor::visit(For &node) {
  SpecialControlFlowGraphVisitor::checkEntrypoint(node);
  std::cout << "Visiting For (" << node.getUniqueNodeId() << ")" << std::endl;
  GraphNode &graphNode = createGraphNodeAndAppendToCfg(node);

  ScopedVisitor::enterScope(node);

  // initializer (e.g., int i = 0;)
  ScopedVisitor::visitChildren(node.getInitializer());
  auto lastStatementInInitializer = lastCreatedNodes;

  // condition expression (e.g., i <= N)
  node.getCondition().accept(*this);
  storeAccessedVariables(graphNode);
  auto lastStatementCondition = lastCreatedNodes;

  // body (e.g., For (-; -; -) { body statements ... })
  node.getBody().accept(*this);
  auto lastStatementInBody = lastCreatedNodes;

  // update statement (e.g., i=i+1;)
  ScopedVisitor::visitChildren(node.getUpdate());
  auto lastStatementInUpdate = lastCreatedNodes;

  ScopedVisitor::exitScope(node);

  auto firstConditionStatement = lastStatementInInitializer.front().get().getControlFlowGraph().getChildren().front();
  if (lastStatementInUpdate.size() > 1) {
    throw std::runtime_error("More than one 'lastStatementInUpdate' in For loop detected. Cannot be handled yet!");
  } else if (lastStatementInUpdate.size()==1) {
    // create an edge in CFG from last update statement to first statement in condition as the condition is checked
    // after executing the update statement of a loop iteration
    lastStatementInUpdate.front().get().getControlFlowGraph().addChild(firstConditionStatement);
  } else if (lastStatementInUpdate.empty()) {
    // if there is no update statement, create an edge from last body statement to condition
    lastStatementInBody.front().get().getControlFlowGraph().addChild(firstConditionStatement);
  }


  // TODO: Fix issue with missing Condition in CFG. Note that Condition must be connected to both
  //  the next statement and the first body statement.
}

void SpecialControlFlowGraphVisitor::visit(Function &node) {
  SpecialControlFlowGraphVisitor::checkEntrypoint(node);
  std::cout << "Visiting Function (" << node.getUniqueNodeId() << ")" << std::endl;
  GraphNode &graphNode = createGraphNodeAndAppendToCfg(node);
  ScopedVisitor::visit(node);
  storeAccessedVariables(graphNode);
}

void SpecialControlFlowGraphVisitor::visit(If &node) {
  SpecialControlFlowGraphVisitor::checkEntrypoint(node);
  std::cout << "Visiting If (" << node.getUniqueNodeId() << ")" << std::endl;
  GraphNode &graphNode = createGraphNodeAndAppendToCfg(node);
  auto lastStatementIf = lastCreatedNodes;

  ScopedVisitor::enterScope(node);

  // condition
  node.getCondition().accept(*this);
  storeAccessedVariables(graphNode);

  // then branch: connect this statement with then branch
  node.getThenBranch().accept(*this);
  auto lastStatementThenBranch = lastCreatedNodes;

  // if existing, connect If statement with Else block
  if (node.hasElseBranch()) {
    lastCreatedNodes = lastStatementIf;
    // else branch
    node.getElseBranch().accept(*this);
  }

  // TODO check if condition is connected with then and also consecutive statement

  ScopedVisitor::exitScope(node);
}

void SpecialControlFlowGraphVisitor::visit(Return &node) {
  SpecialControlFlowGraphVisitor::checkEntrypoint(node);
  std::cout << "Visiting Return (" << node.getUniqueNodeId() << ")" << std::endl;
  GraphNode &graphNode = createGraphNodeAndAppendToCfg(node);
  ScopedVisitor::visit(node);
  storeAccessedVariables(graphNode);
}

void SpecialControlFlowGraphVisitor::visit(VariableDeclaration &node) {
  SpecialControlFlowGraphVisitor::checkEntrypoint(node);
  std::cout << "Visiting VariableDeclaration (" << node.getUniqueNodeId() << ")" << std::endl;
  GraphNode &graphNode = createGraphNodeAndAppendToCfg(node);

  // We do not use ScopedVisitor::visit here as this would visit all children, including the Variable on the left-hand
  // side of the VariableDeclaration. This would register this variable as having been read, although it is written to
  // this variable.

  if (node.hasValue()) {
    node.getValue().accept(*this);
  }

  getCurrentScope().addIdentifier(node.getTarget().getIdentifier());

  if (getCurrentScope().identifierExists(node.getTarget().getIdentifier()) || !ignoreNonDeclaredVariables) {
    markVariableAccess(getCurrentScope().resolveIdentifier(node.getTarget().getIdentifier()),
                       VariableAccessType::WRITE);
  }

  storeAccessedVariables(graphNode);
}

GraphNode &SpecialControlFlowGraphVisitor::createGraphNodeAndAppendToCfg(AbstractStatement &statement) {
  return createGraphNodeAndAppendToCfg(statement, lastCreatedNodes);
}

GraphNode &SpecialControlFlowGraphVisitor::createGraphNodeAndAppendToCfg(
    AbstractStatement &statement,
    const std::vector<std::reference_wrapper<GraphNode>> &parentNodes) {

  // create a new GraphNode for the given statement
  auto graphNodeUniquePtr = std::make_unique<GraphNode>(statement);
  GraphNode &graphNodeRef = *graphNodeUniquePtr;

  // add this node as child to all parents passed in parentNodes vector
  for (auto &p : parentNodes) {
    p.get().getControlFlowGraph().addChild(graphNodeRef);
  }

  // update the lastCreatedNodes vector
  lastCreatedNodes.clear();
  lastCreatedNodes.emplace_back(graphNodeRef);

  // store this node so that we can delete it during the visitor's destruction
  // NOTE: After this statement graphNodeUniquePtr is not accessible anymore!
  nodes.emplace_back(std::move(graphNodeUniquePtr));

  return graphNodeRef;
}

void SpecialControlFlowGraphVisitor::storeAccessedVariables(GraphNode &graphNode) {
  // expects that exactly one GraphNode was created before calling this method
  if (!nodes.empty()) {
    graphNode.setAccessedVariables(std::move(variableAccesses));
  } else {
    throw std::runtime_error("Cannot assign accessed variables to GraphNode because no GraphNode exists.");
  }
}

void SpecialControlFlowGraphVisitor::visit(Variable &node) {
  SpecialControlFlowGraphVisitor::checkEntrypoint(node);
  ScopedVisitor::visit(node);
  if (getCurrentScope().identifierExists(node.getIdentifier()) || !ignoreNonDeclaredVariables) {
    markVariableAccess(getCurrentScope().resolveIdentifier(node.getIdentifier()), VariableAccessType::READ);
  }
}

void SpecialControlFlowGraphVisitor::markVariableAccess(const ScopedIdentifier &scopedIdentifier,
                                                        VariableAccessType accessType) {
  if (variableAccesses.count(scopedIdentifier) > 0) {
    // we potentially need to merge the existing access type and the new one
    if (variableAccesses.at(scopedIdentifier)==accessType) {
      // no need to merge types as both types are the same
      return;
    } else {
      // as they are different and there are only two types (read, write), both (read+write) must be set
      variableAccesses[scopedIdentifier] = VariableAccessType::READ_AND_WRITE;
    }
  } else {
    // we can just add the new variable access
    variableAccesses[scopedIdentifier] = accessType;
  }
}

GraphNode &SpecialControlFlowGraphVisitor::getRootNode() {
  return *nodes.front().get();
}

const GraphNode &SpecialControlFlowGraphVisitor::getRootNode() const {
  return *nodes.front().get();
}

SpecialControlFlowGraphVisitor::SpecialControlFlowGraphVisitor(bool ignoreNonDeclaredVariables)
    : ignoreNonDeclaredVariables(ignoreNonDeclaredVariables) {
}

SpecialControlFlowGraphVisitor::SpecialControlFlowGraphVisitor(
    std::vector<std::string> &alreadyDeclaredVariables) {
  ScopedVisitor::setPredeclaredVariables(alreadyDeclaredVariables);
}


