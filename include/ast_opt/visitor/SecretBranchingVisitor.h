#ifndef GRAPHNODE_H_SRC_VISITOR_SECRETBRANCHINGVISITOR_H_
#define GRAPHNODE_H_SRC_VISITOR_SECRETBRANCHINGVISITOR_H_

#include "ast_opt/ast/AbstractExpression.h"
#include "ast_opt/visitor/ScopedVisitor.h"
#include "TypeCheckingVisitor.h"

// Forward declaration
class SpecialSecretBranchingVisitor;

/// ControlFlowGraphVisitor uses the Visitor<T> template to allow specifying default behaviour
typedef Visitor<SpecialSecretBranchingVisitor> SecretBranchingVisitor;

/// value must be a ptr as variable can be initialized but without a value
typedef std::unordered_map<ScopedIdentifier, AbstractExpression *> VariableValueMap;

class SpecialSecretBranchingVisitor : public ScopedVisitor {
 private:
  bool unsupportedBodyStatementVisited = false;

  VariableValueMap expressionValues;

  SecretTaintedNodesMap &secretTaintedNodesMap;

 public:

  explicit SpecialSecretBranchingVisitor(SecretTaintedNodesMap &taintedNodesMap);

  void visit(If &node) override;

  void visit(For &elem) override;

  void visit(Return &elem) override;

  void visit(Assignment &elem) override;

  static VariableValueMap getChangedVariables(const VariableValueMap &baseMap, const VariableValueMap &changedMap);

  void addIdentifiers(Scope &scope);

  void visit(VariableDeclaration &node) override;

  void visit(FunctionParameter &node) override;
};

#endif //GRAPHNODE_H_SRC_VISITOR_SECRETBRANCHINGVISITOR_H_
