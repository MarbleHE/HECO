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

void SpecialControlFlowGraphVisitor::visit(Assignment &node) {
  std::cout << "Visiting Assignment..." << std::endl;
  auto graphNode = createGraphNodeAndAppendToCfg(node);

  std::string identifier;
  if (auto variable = dynamic_cast<Variable *>(&node.getTarget())) {
    identifier = variable->getIdentifier();
  } else if (auto indexAccess = dynamic_cast<IndexAccess *>(&node.getTarget())) {
    // TODO implement me: must recursively go tree down and retrieve all variable identifiers
  }
  // TODO: replace Scope() by current scope
  markVariableAccess(getCurrentScope().resolveIdentifier(identifier),
                     VariableAccessType::WRITE);
  ScopedVisitor::visit(node);
  storeAccessedVariables(*graphNode);
}

void SpecialControlFlowGraphVisitor::visit(Block &node) {
  std::cout << "Visiting Block" << std::endl;
  auto graphNode = createGraphNodeAndAppendToCfg(node);
  ScopedVisitor::visit(node);
  storeAccessedVariables(*graphNode);
}

void SpecialControlFlowGraphVisitor::visit(For &node) {
  std::cout << "Visiting For" << std::endl;
  auto graphNode = createGraphNodeAndAppendToCfg(node);

  // initializer (e.g., int i = 0;)
  node.getInitializer().accept(*this);
  auto lastStatementInInitializer = lastCreatedNodes;

  // condition expression (e.g., i <= N)
  node.getCondition().accept(*this);
  storeAccessedVariables(*graphNode);
  auto lastStatementCondition = lastCreatedNodes;

  // body (e.g., For (-; -; -) { body statements ... })
  node.getBody().accept(*this);
  auto lastStatementInBody = lastCreatedNodes;

  // update statement (e.g., i=i+1;)
  node.getUpdate().accept(*this);
  auto lastStatementInUpdate = lastCreatedNodes;

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

  // restore the last created nodes in the condition as those need to be connected to the next statement
  lastCreatedNodes = lastStatementCondition;
}

void SpecialControlFlowGraphVisitor::visit(Function &node) {
  std::cout << "Visiting Function" << std::endl;
  auto graphNode = createGraphNodeAndAppendToCfg(node);
  ScopedVisitor::visit(node);
  storeAccessedVariables(*graphNode);
}

void SpecialControlFlowGraphVisitor::visit(If &node) {
  std::cout << "Visiting If" << std::endl;
  auto graphNode = createGraphNodeAndAppendToCfg(node);

  // condition
  node.getCondition().accept(*this);

  // TODO store variable read/writes into current statement


  // then branch: connect this statement with then branch
  node.getThenBranch().accept(*this);
  auto lastStatementThenBranch = lastCreatedNodes;

  // if existing, visit the else branch
  if (node.hasElseBranch()) {

  } else {

  }



  // else branch
  node.getElseBranch().accept(*this);

//  ScopedVisitor::visit(node);
}

void SpecialControlFlowGraphVisitor::visit(Return &node) {
  std::cout << "Visiting Return" << std::endl;

//  ScopedVisitor::visit(node);
}

void SpecialControlFlowGraphVisitor::visit(VariableDeclaration &node) {
  std::cout << "Visiting VariableDeclaration" << std::endl;

//  ScopedVisitor::visit(node);
}

GraphNode *SpecialControlFlowGraphVisitor::createGraphNodeAndAppendToCfg(AbstractStatement &statement) {
  createGraphNodeAndAppendToCfg(statement, lastCreatedNodes);
}

GraphNode *SpecialControlFlowGraphVisitor::createGraphNodeAndAppendToCfg(
    AbstractStatement &statement,
    const std::vector<std::reference_wrapper<GraphNode>> &parentNodes) {

  // create a new GraphNode for the given statement
  auto graphNode = std::make_unique<GraphNode>(statement);

  // add this node as child to all parents passed in parentNodes vector
  for (auto &p : parentNodes) {
    p.get().getControlFlowGraph().addChild(*graphNode);
  }

  // store this node so that we can delete it during the visitor's destruction
  nodes.emplace_back(std::move(graphNode));

  // update the lastCreatedNodes vector
  lastCreatedNodes.clear();
  lastCreatedNodes.emplace_back(**nodes.end());
}

SpecialControlFlowGraphVisitor::~SpecialControlFlowGraphVisitor() {
  // delete all nodes belonging to this ControlFlowGraphVisitor (CFG/DFG)
  nodes.clear();
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
  // TODO: replace Scope() by current scope
  markVariableAccess(getCurrentScope().resolveIdentifier(node.getIdentifier()),
                     VariableAccessType::READ);
}

void SpecialControlFlowGraphVisitor::markVariableAccess(const ScopedIdentifier &scopedIdentifier,
                                                        VariableAccessType accessType) {
  variableAccesses.emplace(scopedIdentifier, accessType);
}

GraphNode &SpecialControlFlowGraphVisitor::getRootNode() {
  return *nodes.front().get();
}

const GraphNode &SpecialControlFlowGraphVisitor::getRootNode() const {
  return *nodes.front().get();
}
