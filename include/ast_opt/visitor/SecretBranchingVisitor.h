#ifndef GRAPHNODE_H_SRC_VISITOR_SECRETBRANCHINGVISITOR_H_
#define GRAPHNODE_H_SRC_VISITOR_SECRETBRANCHINGVISITOR_H_

#include "ast_opt/visitor/ScopedVisitor.h"

// Forward declaration
class SpecialSecretBranchingVisitor;

/// ControlFlowGraphVisitor uses the Visitor<T> template to allow specifying default behaviour
typedef Visitor<SpecialSecretBranchingVisitor> SecretBranchingVisitor;

class SpecialSecretBranchingVisitor : public ScopedVisitor {
 private:
  bool unsupportedBodyStatementVisited = false;

 public:
  void visit(If &node) override;

  void visit(For &elem) override;

  void visit(Return &elem) override;
};

#endif //GRAPHNODE_H_SRC_VISITOR_SECRETBRANCHINGVISITOR_H_
