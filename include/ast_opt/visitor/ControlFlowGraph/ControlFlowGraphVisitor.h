#ifndef AST_OPTIMIZER_SRC_VISITOR_CONTROLFLOWGRAPHVISITOR_H_
#define AST_OPTIMIZER_SRC_VISITOR_CONTROLFLOWGRAPHVISITOR_H_

#include <iostream>
#include <set>
#include <ast_opt/ast/AbstractStatement.h>
#include "ast_opt/visitor/ScopedVisitor.h"
#include "ast_opt/visitor/ControlFlowGraph/GraphNode.h"

// Forward declaration
class SpecialControlFlowGraphVisitor;

/// ControlFlowGraphVisitor uses the Visitor<T> template to allow specifying default behaviour
typedef Visitor<SpecialControlFlowGraphVisitor> ControlFlowGraphVisitor;

class SpecialControlFlowGraphVisitor : public ScopedVisitor {
 private:

  // A list of all GraphNodes belonging to this CFG/DFG and created by this ControlFlowGraphVisitor.
  // Node at index 0 is the root node.
  std::vector<std::unique_ptr<GraphNode>> nodes;

  /// The nodes that were created most recently. Those are the parent nodes of the next node to be created.
  std::vector<std::reference_wrapper<GraphNode>> lastCreatedNodes;

  GraphNode *createGraphNodeAndAppendToCfg(AbstractStatement &statement);

  GraphNode *createGraphNodeAndAppendToCfg(AbstractStatement &statement,
                                           const std::vector<std::reference_wrapper<GraphNode>> &parentNodes);

  /// A set containing pairs of (variable identifier, access type) where variable identifier is the name of a variable
  /// and access type describes if a variable was read or written. This set collects all information of visited children
  /// of a statement and is cleared before leaving a statement (see postActionsStatementVisited).
  std::set<VariableAccessPair> variableAccesses;

 public:

  virtual ~SpecialControlFlowGraphVisitor();

  void visit(Assignment &node) override;

  void visit(Block &node) override;

  void visit(For &node) override;

  void visit(Function &node) override;

  void visit(If &node) override;

  void visit(Return &node) override;

  void visit(Variable &node) override;

  void visit(VariableDeclaration &node) override;

  void storeAccessedVariables(GraphNode &graphNode);

  void markVariableAccess(const ScopedIdentifier &scopedIdentifier, VariableAccessType accessType);

  [[nodiscard]] GraphNode &getRootNode();

  [[nodiscard]] const GraphNode &getRootNode() const;
};

#endif //AST_OPTIMIZER_SRC_VISITOR_CONTROLFLOWGRAPHVISITOR_H_
