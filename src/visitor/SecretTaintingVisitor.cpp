#include "SecretTaintingVisitor.h"
#include "FunctionParameter.h"
#include "VarAssignm.h"
#include "VarDecl.h"
#include "ArithmeticExpr.h"
#include "Return.h"
#include "Block.h"
#include "Variable.h"
#include "LogicalExpr.h"
#include "Function.h"
#include "Call.h"

void SecretTaintingVisitor::visit(Ast &elem) {
  Visitor::visit(elem);
}

void SecretTaintingVisitor::visit(Call &elem) {
  Visitor::visit(elem);
  // after visiting the Call's referenced Function, check if the Function node was tainted
  // if the Function was tainted -> taint the Call node too
  if (elem.getFunc()!=nullptr && nodeIsTainted(*elem.getFunc())) {
    addTaintedNode(elem.AbstractExpr::castTo<AbstractNode>());
  }
}

void SecretTaintingVisitor::visit(CallExternal &elem) {
  throw std::invalid_argument("ASTs containing CallExternal objects are not supported by the SecretTaintingVisitor!");
  //  Visitor::visit(elem);
}

// ==========================
// STATEMENTS
// ==========================

void SecretTaintingVisitor::visit(Block &elem) {
  Visitor::visit(elem);
  // if after processing all the Block's children, any of them are tainted, this block also will be tainted
  auto statements = elem.getStatements();
  auto statementsAsNodes = std::vector<AbstractNode *>(statements.begin(), statements.end());
  if (anyNodesAreTainted(statementsAsNodes))
    addTaintedNode(elem.castTo<AbstractNode>());
}

void SecretTaintingVisitor::visit(Function &elem) {
  Visitor::visit(elem);
  // if after processing the Function's body, any of the function body's statements are tainted, this Function also will
  // be tainted (this makes sense, e.g., if Function is part of a called sub-function that is referenced by a Call obj.)
  auto bodyStatements = elem.getBodyStatements();
  auto statementsAsNodes = std::vector<AbstractNode *>(bodyStatements.begin(), bodyStatements.end());
  if (anyNodesAreTainted(statementsAsNodes)) addTaintedNode(elem.castTo<AbstractNode>());
}

void SecretTaintingVisitor::visit(If &elem) {
  throw std::invalid_argument("ASTs containing If statements are not supported by the SecretTaintingVisitor!");
  //  Visitor::visit(elem);
}

void SecretTaintingVisitor::visit(Return &elem) {
  std::set<std::string> returnValueVariables;
  // go through all expression of the return statement and collected the variables contained therein
  for (auto &curExpression : elem.getReturnExpressions()) {
    // collect all variables in the current expression
    auto vars = curExpression->getVariableIdentifiers();
    // copy the variables into the set returnValueVariables
    std::copy(vars.begin(), vars.end(), std::inserter(returnValueVariables, returnValueVariables.end()));
  }
  // update tainted status
  if (nodeIsTainted(elem)
      || anyVariableIdentifierIsTainted(returnValueVariables.begin(), returnValueVariables.end())) {
    // mark current node as tainted (as taintedNodes is a set, adding it again does not change anything)
    taintedNodes.insert(elem.getUniqueNodeId());
    // mark all of its descendants as tainted
    addTaintedNodes(elem.getDescendants());
    // mark all variables as tainted
    addTaintedVariableIdentifiers(returnValueVariables.begin(), returnValueVariables.end());
  }
  Visitor::visit(elem);
}

void SecretTaintingVisitor::visit(While &elem) {
  throw std::invalid_argument("ASTs containing While objects are not supported by the SecretTaintingVisitor!");
//  Visitor::visit(elem);
}

void SecretTaintingVisitor::visit(VarAssignm &elem) {
  checkAndAddTaintedChildren(static_cast<AbstractStatement *>(&elem), elem.getValue()->getVariableIdentifiers());
  Visitor::visit(elem);
  // after visiting the initializer, check if it is tainted - this is needed for Call nodes
  if (nodeIsTainted(*elem.getValue())) {
    addTaintedNode(&elem);
  }
}

void SecretTaintingVisitor::visit(VarDecl &elem) {
  if (elem.getDatatype()->isEncrypted()) {
    taintedVariables.insert(elem.getVarTargetIdentifier());
    taintedNodes.insert(elem.getUniqueNodeId());
    addTaintedNodes(elem.getDescendants()); // TODO check why Node_ is enqueue here
  }
  // this does not consider Call nodes
  checkAndAddTaintedChildren(static_cast<AbstractStatement *>(&elem), elem.getInitializer()->getVariableIdentifiers());
  Visitor::visit(elem);
  // after visiting the initializer, check if it is tainted - this is needed for Call nodes
  if (nodeIsTainted(*elem.getInitializer())) {
    addTaintedNode(&elem);
  }
}

// ==========================
// EXPRESSIONS
// ==========================

void SecretTaintingVisitor::visit(ArithmeticExpr &elem) {
  Visitor::visit(elem);
}

void SecretTaintingVisitor::visit(FunctionParameter &elem) {
  // if this FunctionParameter refers to an encrypted variable -> all of its variable identifiers are tainted
  if (elem.getDatatype()->isEncrypted()) {
    // taint the FunctionParameter object
    addTaintedNode(&elem);
    // remember the Variable identifiers associated
    auto varIdentifiers = elem.getVariableIdentifiers();
    addTaintedVariableIdentifiers(varIdentifiers.begin(), varIdentifiers.end());
    // taint the Variable objects
    for (auto &child : elem.getChildrenNonNull()) {
      auto childAsVariable = dynamic_cast<Variable *>(child);
      if (childAsVariable!=nullptr
          && std::count(varIdentifiers.begin(), varIdentifiers.end(), childAsVariable->getIdentifier()) > 0) {
        addTaintedNode(child);
      }
    }
  }
  Visitor::visit(elem);
}

void SecretTaintingVisitor::visit(LiteralBool &elem) {
  Visitor::visit(elem);
}

void SecretTaintingVisitor::visit(LiteralInt &elem) {
  Visitor::visit(elem);
}

void SecretTaintingVisitor::visit(LiteralString &elem) {
  Visitor::visit(elem);
}

void SecretTaintingVisitor::visit(LiteralFloat &elem) {
  Visitor::visit(elem);
}

void SecretTaintingVisitor::visit(LogicalExpr &elem) {
  Visitor::visit(elem);
}

void SecretTaintingVisitor::visit(Operator &elem) {
  Visitor::visit(elem);
}

void SecretTaintingVisitor::visit(UnaryExpr &elem) {
  Visitor::visit(elem);
}

void SecretTaintingVisitor::visit(Datatype &elem) {
  if (elem.isEncrypted()) addTaintedNode(&elem);
  Visitor::visit(elem);
}

void SecretTaintingVisitor::visit(Variable &elem) {
  if (nodeIsTainted(elem)) taintedVariables.insert(elem.getIdentifier());
  Visitor::visit(elem);
}



// ==========================
// HELPER METHODS
// ==========================

const std::set<std::string> &SecretTaintingVisitor::getSecretTaintingList() const {
  return taintedNodes;
}

bool SecretTaintingVisitor::nodeIsTainted(AbstractNode &node) const {
  return taintedNodes.count(node.getUniqueNodeId()) > 0;
}

bool SecretTaintingVisitor::anyNodesAreTainted(std::vector<AbstractNode *> nodes) const {
  return std::any_of(nodes.begin(), nodes.end(), [&](AbstractNode *n) {
    return taintedNodes.count(n->getUniqueNodeId()) > 0;
  });
}

void SecretTaintingVisitor::addTaintedNodes(std::vector<AbstractNode *> nodesToAdd) {
  // copy all unique node IDs of nodesToAdd to taintedNodes
  std::transform(nodesToAdd.begin(), nodesToAdd.end(), std::inserter(taintedNodes, taintedNodes.end()),
                 [](AbstractNode *n) -> std::string {
                   return n->getUniqueNodeId();
                 });
}

void SecretTaintingVisitor::addTaintedNode(AbstractNode *n) {
  taintedNodes.insert(n->getUniqueNodeId());
}

void SecretTaintingVisitor::checkAndAddTaintedChildren(AbstractStatement *n,
                                                       std::vector<std::string> varIdentifiersInRhs) {
  // if any variable used in the initializer of this elem is tainted -> the whole assignment becomes tainted
  if (anyVariableIdentifierIsTainted(varIdentifiersInRhs.begin(), varIdentifiersInRhs.end())) {
    // add the target variable (e.g., 'alpha' in 'int alpha = secretX + 4')
    taintedVariables.insert(n->getVarTargetIdentifier());
    // mark this node as tainted
    taintedNodes.insert(n->getUniqueNodeId());
    // also all children of this assignment become tainted
    for (auto &node : n->getDescendants()) taintedNodes.insert(node->getUniqueNodeId());
  }
}
