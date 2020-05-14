#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_CONTROLFLOWGRAPHVISITOR_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_CONTROLFLOWGRAPHVISITOR_H_

#include <utility>
#include <vector>
#include <string>
#include <iostream>
#include <set>
#include <deque>
#include <stack>
#include <unordered_set>
#include "Visitor.h"

// forward declarations
class GraphNode;

/// An enum class to describe how a variable was accessed. Supported values are READ and WRITE.
enum class AccessType { READ = 0, WRITE = 1 };

/// This class constructs the Control Flow Graph (CFG) that represents the program's control flow. Note that the
/// condition of the AST objects "For", "If", "While" are considered as statements too, although our AST treats them as
/// expressions (AbstractExpr). This is required to model the data flow of variables in the condition.
/// The visitor further collects information that is required to build the Data Flow Graph (DFG) therefrom.
class ControlFlowGraphVisitor : public Visitor {
 private:
  /// Ugly hack: Variables in CFG that are both read and written to
  std::set<std::string> variablesReadAndWritten;

  /// The nodes that were created most recently. Those are the parent nodes of the next node to be created.
  std::vector<GraphNode *> lastCreatedNodes;

  /// Defines the access mode to be used by appendStatementToCfg if no access mode is explicitly given. This is needed
  /// because in certain cases only the parent node knows if the subsequent Variable nodes are writes or reads.
  /// NOTE: The defaultAccessMode must be set back to AccessType::READ after visiting the children.
  AccessType defaultAccessMode{AccessType::READ};

  /// A set containing pairs of (variable identifier, access type) where variable identifier is the name of a variable
  /// and access type describes if a variable was read or written. This set collects all information of visited children
  /// of a statement and is cleared before leaving a statement (see postActionsStatementVisited).
  std::set<std::pair<std::string, AccessType>> varAccess;

  /// The root node of the control flow graph that also contains information about the data flow.
  GraphNode *rootNode;

  /// A flag that defines whether visited expressions should be treated as statements and hence included into the
  /// control flow graph. This is a workaround for the condition (AbstractExpr) in If, For, While objects because their
  /// condition and accessed variables would otherwise not be present in the control/data flow graph.
  bool handleExpressionsAsStatements{false};

 public:
  /// Marks a variable as accessed by using the value in defaultAccessMode.
  /// \param var The variable to be marked as accessed.
  void markVariableAccess(Variable &var);

  /// Marks a variable as read or write, depending on the given access type.
  /// \param variableIdentifier The variable's identifier ("variable name").
  /// \param accessMode Whether the variable access should be a read or write.
  void markVariableAccess(const std::string &variableIdentifier, AccessType accessMode);

  /// Creates a new node for the control flow graph of type GraphNode. Adds the created node as child of each of the
  /// nodes in lastCreatedNodes, and returns the created GraphNode.
  /// \param node The node (e.g., AbstractStatement) to be added to the control flow graph.
  /// \return A pointer to the GraphNode created based on the given AbstractNode.
  GraphNode *appendStatementToCfg(AbstractNode &node);

  /// Creates a new node for the control flow graph of type GraphNode. Adds the created node as child of each of the
  /// given nodes in parentNodes, and returns the created GraphNode.
  /// \param node The node (e.g., AbstractStatement) to be added to the control flow graph.
  /// \param parentNodes The nodes to be used as parent node of the newly created GraphNode.
  /// \return A pointer to the GraphNode created based on the given AbstractNode.
  GraphNode *appendStatementToCfg(AbstractNode &abstractStmt, const std::vector<GraphNode *> &parentNodes);

  /// Returns the root node of the control flow graph, i.e., the node that represents the entry point of the program and
  /// does not have any parent nodes.
  /// \return The root node of the control flow graph.
  [[nodiscard]] GraphNode *getRootNodeCfg() const;

  /// A method that must be called after finishing the visit of a statement. The method either moves the variables
  /// that were accessed in the statement's children to the given GraphNode (if gNode != nullptr), which is the node
  /// that was created for the current statement, or clears only the accessed variables (if gNode == nullptr). After
  /// calling postActionsStatementVisited the lastCreatedNodes vector is always empty.
  /// \param gNode The node where the recently accessed variables should be moved to.
  /// \invariant lastCreatedNodes is always empty after calling this method.
  void postActionsStatementVisited(GraphNode *gNode);

  /** @defgroup visit Methods for handling visits of AbstractNode subclasses
   *  @{
   */
  void visit(ArithmeticExpr &elem) override;

  void visit(Block &elem) override;

  void visit(Call &elem) override;

  void visit(CallExternal &elem) override;

  void visit(Datatype &elem) override;

  void visit(For &elem) override;

  void visit(Function &elem) override;

  void visit(FunctionParameter &elem) override;

  void visit(If &elem) override;

  void visit(LiteralBool &elem) override;

  void visit(LiteralInt &elem) override;

  void visit(LiteralString &elem) override;

  void visit(LiteralFloat &elem) override;

  void visit(LogicalExpr &elem) override;

  void visit(Operator &elem) override;

  void visit(ParameterList &elem) override;

  void visit(Return &elem) override;

  void visit(Rotate &elem) override;

  void visit(Transpose &elem) override;

  void visit(UnaryExpr &elem) override;

  void visit(VarAssignm &elem) override;

  void visit(VarDecl &elem) override;

  void visit(Variable &elem) override;

  void visit(While &elem) override;

  void visit(MatrixElementRef &elem) override;

  void visit(Ast &elem) override;

  void visit(MatrixAssignm &elem) override;

  void visit(OperatorExpr &elem) override;

  void visit(GetMatrixSize &elem) override;
  /** @} */ // End of visit group

  void buildDataFlowGraph();

  /// Return all variables that were both written and read in the current CFG
  /// WARNING: only returns valid answers when Data Flow Graph has been built!
  /// \return all variables that are both written and read in the current CFG
  std::set<std::string> getVariablesReadAndWritten();

  void handleOperatorExpr(AbstractExpr &ae);
};

#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_CONTROLFLOWGRAPHVISITOR_H_
