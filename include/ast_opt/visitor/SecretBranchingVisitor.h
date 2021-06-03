#ifndef GRAPHNODE_H_SRC_VISITOR_SECRETBRANCHINGVISITOR_H_
#define GRAPHNODE_H_SRC_VISITOR_SECRETBRANCHINGVISITOR_H_

#include "ast_opt/ast/AbstractExpression.h"
#include "ast_opt/utilities/Visitor.h"
#include "TypeCheckingVisitor.h"
#include "ast_opt/utilities/VariableMap.h"

// Forward declaration
class SpecialSecretBranchingVisitor;

/// ControlFlowGraphVisitor uses the Visitor<T> template to allow specifying default behaviour
typedef Visitor<SpecialSecretBranchingVisitor> SecretBranchingVisitor;

/// value must be a ptr as variable can be declared only (i.e., uninitialized)
typedef VariableMap<AbstractExpression*> VariableValueMap;

class SpecialSecretBranchingVisitor : public ScopedVisitor {
 private:
  /// A map that stores the current value, while traversing the AST, for each scoped identifier.
  VariableValueMap expressionValues;

  /// The map (passed in constructor) that provides information about expression nodes involving secret variables.
  SecretTaintedNodesMap &secretTaintedNodesMap;

  /// A flag that indicates whether the body of an If statement included an unsupported statement (e.g., For, Return)
  /// that inhibits rewriting this If statement.
  bool unsupportedBodyStatementVisited = false;

  /// A flag that helps to propagate the information that an If statement can be deleted to its parent node
  /// (i.e., a Block statement), which then executes the deletion.
  bool visitedStatementMarkedForDeletion = false;

  /// The statements to be placed at the position where the node marked for deletion was before
  /// (see visitedStatementMarkedForDeletion).
  std::vector<std::unique_ptr<AbstractStatement>> replacementStatements;

 public:
  /// Creates a new SpecialSecretBranchingVisitor by taking a map of <unique node ID, bool> as input that provides
  /// information about expressions dealing with secret variables.
  /// \param taintedNodesMap A map of <unique node ID, bool> containing information about secret tainted nodes.
  explicit SpecialSecretBranchingVisitor(SecretTaintedNodesMap &taintedNodesMap);

  /// Compares each identifier in changedMap with the identifiers in baseMap. Returns the identifiers that either have
  /// a different value or do not exist in baseMap at all.
  /// \param baseMap The base map to compare the identifiers from changedMap with.
  /// \param changedMap The changed map whose identifiers are compared with those provided by baseMap.
  /// \return (A VariableValueMap) that contains all identifiers with a changed value or are not present in baseMap.
  static VariableValueMap getChangedVariables(const VariableValueMap &baseMap, const VariableValueMap &changedMap);

  void visit(If &node) override;

  void visit(For &elem) override;

  void visit(Return &elem) override;

  void visit(Assignment &elem) override;

  void visit(VariableDeclaration &node) override;

  void visit(FunctionParameter &node) override;

  void visit(Block &node) override;
};

#endif //GRAPHNODE_H_SRC_VISITOR_SECRETBRANCHINGVISITOR_H_
