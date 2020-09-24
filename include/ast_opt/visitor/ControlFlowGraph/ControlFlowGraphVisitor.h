#ifndef AST_OPTIMIZER_SRC_VISITOR_CONTROLFLOWGRAPHVISITOR_H_
#define AST_OPTIMIZER_SRC_VISITOR_CONTROLFLOWGRAPHVISITOR_H_

#include <iostream>
#include <set>
#include "ast_opt/ast/AbstractStatement.h"
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

  /// Creates a new GraphNode for the given AbstractStatement node and appends it as child to each of the last created
  /// GraphNodes (see lastCreatedNodes). This method only accepts an AbstractStatement because the CFG/DFG only consider
  /// statements.
  /// \param statement The statement for which a new GraphNode object should be created for.
  /// \return (A reference to) the newly created GraphNode object.
  GraphNode &createGraphNodeAndAppendToCfg(AbstractStatement &statement);

  /// This constructor is required for testing purposes.
  /// Creates a new GraphNode for the given AbstractStatement node and appends it as child to each of the last created
  /// GraphNodes (see lastCreatedNodes). This method only accepts an AbstractStatement because the CFG/DFG only consider
  /// statements.
  /// \param statement The statement for which a new GraphNode object should be created for.
  /// \param parentNodes The node(s) that should be added as parent nodes for the newly created GraphNode.
  /// \return (A reference to) the newly created GraphNode object.
  GraphNode &createGraphNodeAndAppendToCfg(AbstractStatement &statement,
                                           const std::vector<std::reference_wrapper<GraphNode>> &parentNodes);

  /// A set containing pairs of (scoped identifiers, access type) where the access type describes if a variable was read
  /// or written. This set collects all information of visited children of a statement and must be cleared before
  /// leaving a statement by calling storeAccessedVariables.
  std::set<VariableAccessPair> variableAccesses;

 public:
  virtual ~SpecialControlFlowGraphVisitor() = default;

  void visit(Assignment &node) override;

  void visit(Block &node) override;

  void visit(For &node) override;

  void visit(Function &node) override;

  void visit(If &node) override;

  void visit(Return &node) override;

  void visit(Variable &node) override;

  void visit(VariableDeclaration &node) override;

  /// Stores the tracked variable accesses in the given GraphNode and clears the variable access map used for tracking.
  /// This assigns the variable access information, collected by visiting a statement's children nodes, to the
  /// corresponding GraphNode in the CFG.
  /// \param graphNode The GraphNode to that the tracked variable accessed should be assigned to.
  void storeAccessedVariables(GraphNode &graphNode);

  /// Marks a scoped identifier (e.g., Variable) as accessed by the given access type.
  /// \param scopedIdentifier The identifier that should be marked as accessed.
  /// \param accessType The access type that describes how the variable was accessed (read or write).
  void markVariableAccess(const ScopedIdentifier &scopedIdentifier, VariableAccessType accessType);

  /// Gets the root node of the control flow graph.
  /// \return (A reference to) the root node of the control flow graph.
  [[nodiscard]] GraphNode &getRootNode();

  /// Gets the root node of the control flow graph.
  /// \return (A const reference to) the root node of the control flow graph.
  [[nodiscard]] const GraphNode &getRootNode() const;

  /// Checks whether the given node is supported to be the first node where the ControlFlowGraphVisitor is called on.
  /// The ControlFlowGraphVisitor requires this to be a Block, For, of If node.
  /// \param node The node on that visit(...) was called on.
  void checkEntrypoint(AbstractNode &node);
};

#endif //AST_OPTIMIZER_SRC_VISITOR_CONTROLFLOWGRAPHVISITOR_H_
