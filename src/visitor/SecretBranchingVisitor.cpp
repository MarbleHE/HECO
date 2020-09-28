#include "ast_opt/visitor/SecretBranchingVisitor.h"
#include "ast_opt/ast/If.h"

void SpecialSecretBranchingVisitor::visit(If &) {
  unsupportedBodyStatementVisited = false;

  // visit the if statement's condition and body

  if (!unsupportedBodyStatementVisited) {
    // do rewriting
  }
}

void SpecialSecretBranchingVisitor::visit(For &node) {
  unsupportedBodyStatementVisited = true;
  ScopedVisitor::visit(node);
}

void SpecialSecretBranchingVisitor::visit(Return &node) {
  unsupportedBodyStatementVisited = true;
  ScopedVisitor::visit(node);
}

