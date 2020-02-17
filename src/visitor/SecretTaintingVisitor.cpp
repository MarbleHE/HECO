#include "SecretTaintingVisitor.h"
#include "FunctionParameter.h"
#include "VarAssignm.h"
#include "VarDecl.h"

void SecretTaintingVisitor::visit(Ast &elem) {
  Visitor::visit(elem);
}

void SecretTaintingVisitor::visit(BinaryExpr &elem) {
  Visitor::visit(elem);
}
void SecretTaintingVisitor::visit(Block &elem) {
  Visitor::visit(elem);
}
void SecretTaintingVisitor::visit(Call &elem) {
  Visitor::visit(elem);
}
void SecretTaintingVisitor::visit(CallExternal &elem) {
  Visitor::visit(elem);
}
void SecretTaintingVisitor::visit(Function &elem) {
  Visitor::visit(elem);
}
void SecretTaintingVisitor::visit(FunctionParameter &elem) {
  if (elem.getDatatype()->isEncrypted()) {
    for (auto &var : elem.getVariableIdentifiers()) this->taintedVariables.push_back(var);
  }
  Visitor::visit(elem);
}
void SecretTaintingVisitor::visit(If &elem) {
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
void SecretTaintingVisitor::visit(Return &elem) {
  Visitor::visit(elem);
}
void SecretTaintingVisitor::visit(UnaryExpr &elem) {
  Visitor::visit(elem);
}
void SecretTaintingVisitor::visit(VarAssignm &elem) {
  auto varIdentifiersInRhs = elem.getValue()->getVariableIdentifiers();
  auto identifierIsTainted = [&](const std::string &identifier) {
    return std::find(taintedVariables.begin(), taintedVariables.end(), identifier)!=taintedVariables.end();
  };
  // if any variable used in the initializer of this variable assignment is tainted -> the whole assignment becomes
  // tainted
  if (std::any_of(varIdentifiersInRhs.begin(), varIdentifiersInRhs.end(), identifierIsTainted)) {
    taintedVariables.push_back(elem.getVarTargetIdentifier());
    taintedNodes.push_back(elem.getUniqueNodeId());
    // also all children of this assignment become tainted
    for (auto &node : elem.getDescendants()) taintedNodes.push_back(node->getUniqueNodeId());
  }
  Visitor::visit(elem);
}
void SecretTaintingVisitor::visit(VarDecl &elem) {
  auto varIdentifiersInRhs = elem.getInitializer()->getVariableIdentifiers();
  auto identifierIsTainted = [&](const std::string &identifier) {
    return std::find(taintedVariables.begin(), taintedVariables.end(), identifier)!=taintedVariables.end();
  };
  // if any variable used in the initializer of this variable assignment is tainted -> the whole assignment becomes
  // tainted
  if (std::any_of(varIdentifiersInRhs.begin(), varIdentifiersInRhs.end(), identifierIsTainted)) {
    taintedVariables.push_back(elem.getVarTargetIdentifier());
    taintedNodes.push_back(elem.getUniqueNodeId());
    // also all children of this assignment become tainted
    for (auto &node : elem.getDescendants()) taintedNodes.push_back(node->getUniqueNodeId());
  }
  Visitor::visit(elem);
}
void SecretTaintingVisitor::visit(Variable &elem) {
  if (std::find(taintedNodes.begin(), taintedNodes.end(), elem.getIdentifier())!=taintedNodes.end()) {
    taintedVariables.push_back(elem.getIdentifier());
  }
  Visitor::visit(elem);
}
void SecretTaintingVisitor::visit(While &elem) {
  Visitor::visit(elem);
}
const std::vector<std::string> &SecretTaintingVisitor::getSecretTaintingList() const {
  return taintedNodes;
}
