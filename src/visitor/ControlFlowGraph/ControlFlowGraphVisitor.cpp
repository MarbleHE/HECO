#include "ast_opt/visitor/ControlFlowGraph/ControlFlowGraphVisitor.h"

void SpecialControlFlowGraphVisitor::visit(Assignment &node) {
  std::cout << "Visiting Assignment" << std::endl;
  ScopedVisitor::visit(node);
}

void SpecialControlFlowGraphVisitor::visit(Block &node) {
  std::cout << "Visiting Block" << std::endl;
//  auto graphNode = appendStatementToCfg(node);

  for (auto &c : node) {
    c.accept(*this);
  }
//  postActionsStatementVisited(graphNode);
}

void SpecialControlFlowGraphVisitor::visit(For &node) {
  std::cout << "Visiting For" << std::endl;

  visit(node.getInitializer());

  visit(node.getUpdate());

  visit(node.getBody());

  ScopedVisitor::visit(node);
}

void SpecialControlFlowGraphVisitor::visit(Function &node) {
  std::cout << "Visiting Function" << std::endl;

  // visit the children of the CFG
  visit(node.getBody());

  ScopedVisitor::visit(node);
}

void SpecialControlFlowGraphVisitor::visit(If &node) {
  std::cout << "Visiting If" << std::endl;

  visit(node.getThenBranch());

  visit(node.getElseBranch());

  ScopedVisitor::visit(node);
}

void SpecialControlFlowGraphVisitor::visit(Return &node) {
  std::cout << "Visiting Return" << std::endl;

  ScopedVisitor::visit(node);
}

void SpecialControlFlowGraphVisitor::visit(VariableDeclaration &node) {
  std::cout << "Visiting VariableDeclaration" << std::endl;

  ScopedVisitor::visit(node);
}
