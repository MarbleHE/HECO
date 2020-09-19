#ifndef AST_OPTIMIZER_SRC_VISITOR_CONTROLFLOWGRAPHVISITOR_H_
#define AST_OPTIMIZER_SRC_VISITOR_CONTROLFLOWGRAPHVISITOR_H_

#include <iostream>
#include "ast_opt/visitor/Visitor.h"
#include "ast_opt/ast/Assignment.h"
#include "ast_opt/ast/Block.h"
#include "ast_opt/ast/For.h"
#include "ast_opt/ast/Function.h"
#include "ast_opt/ast/If.h"
#include "ast_opt/ast/Return.h"
#include "ast_opt/ast/VariableDeclaration.h"

// Forward declaration
class SpecialControlFlowGraphVisitor;

/// ControlFlowGraphVisitor uses the Visitor<T> template to allow specifying default behaviour
typedef Visitor<SpecialControlFlowGraphVisitor> ControlFlowGraphVisitor;

class SpecialControlFlowGraphVisitor : public ScopedVisitor {
 private:

 public:
  void visit(Assignment &node) override;

  void visit(Block &node) override;

  void visit(For &node) override;

  void visit(Function &node) override;

  void visit(If &node) override;

  void visit(Return &node) override;

  void visit(VariableDeclaration &node) override;
};

#endif //AST_OPTIMIZER_SRC_VISITOR_CONTROLFLOWGRAPHVISITOR_H_
