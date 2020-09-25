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

  /// Creates a new GraphNode for the given AbstractNode and appends it as child to each of the last created
  /// GraphNodes (see lastCreatedNodes).
  /// \param astNode The node for which a new GraphNode object should be created for.
  /// \return (A reference to) the newly created GraphNode object.
  GraphNode &createGraphNodeAndAppendToCfg(AbstractNode &astNode);

  /// This constructor is required for testing purposes.
  /// Creates a new GraphNode for the given AbstractNode and appends it as child to each of the last created
  /// GraphNodes (see lastCreatedNodes).
  /// \param statement The node for which a new GraphNode object should be created for.
  /// \param parentNodes The node(s) that should be added as parent nodes for the newly created GraphNode.
  /// \return (A reference to) the newly created GraphNode object.
  GraphNode &createGraphNodeAndAppendToCfg(AbstractNode &statement,
                                           const std::vector<std::reference_wrapper<GraphNode>> &parentNodes);

  /// A set containing pairs of (scoped identifiers, access type) where the access type describes if a variable was read
  /// or written. This set collects all information of visited children of a statement and must be cleared before
  /// leaving a statement by calling storeAccessedVariables.
  VarAccessMapType variableAccesses;

  // This class has implemented two ways to deal with variables that are declared in a part of the program
  // that is not in the given sub-AST and as such not visited by the visitor:
  //   1. If the visitor is initialized with ControlFlowGraphVisitor(ignoreNonDeclaredVariables = false), then variables
  // for which no declaration could be found will just be ignored. Consequently, also read/write accesses to them are
  // not being tracked in the variableAccesses map.
  //   2. Alternatively, it is possible to manually pass a std::vector<std::string> in the corresponding constructor
  // with those variables that are declared in another part of the program (i.e., in any parent scope as we always
  // require to pass a scope opener as input node). In that case these variables will be considered as being declared
  // in the outermost scope and accesses will be tracked as usual.

  /// Defines whether non-declared variables are ignored when calling Scope::resolveIdentifier. If this is not enabled
  /// (ignoreNonDeclaredVariables = False), then Scope::resolveIdentifier will raise an exception. Note that if non-
  /// declared variables are ignored, their read/write access is not tracked in the variableAccesses map.
  bool ignoreNonDeclaredVariables = false;

 public:
  SpecialControlFlowGraphVisitor() = default;

  explicit SpecialControlFlowGraphVisitor(bool ignoreNonDeclaredVariables);

  explicit SpecialControlFlowGraphVisitor(std::vector<std::string> &variablesDeclared);

  virtual ~SpecialControlFlowGraphVisitor() = default;

  void visit(Assignment &node) override;

  void visit(Block &node) override;

  void visit(For &node) override;

  void visit(Function &node) override;

  void visit(FunctionParameter &node) override;

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
