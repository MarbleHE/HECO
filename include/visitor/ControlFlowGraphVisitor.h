#ifndef AST_OPTIMIZER_INCLUDE_VISITOR_CONTROLFLOWGRAPHVISITOR_H_
#define AST_OPTIMIZER_INCLUDE_VISITOR_CONTROLFLOWGRAPHVISITOR_H_

#include <utility>
#include <vector>
#include <string>
#include <iostream>
#include <set>
#include <deque>
#include <stack>
#include <unordered_set>
#include "Visitor.h"
#include "AbstractNode.h"
#include "Variable.h"

struct GraphNode;

enum class AccessType { READ, WRITE };

/// This class constructs the Control Flow Graph (CFG) that represents the program's control flow. Note that the
/// condition of the AST objects "For", "If", "While" are considered as statements too, although our AST treats them as
/// expressions (AbstractExpr). This is required to model the data flow of variables in the condition.
/// The visitor further collects information that is required to build the Data Flow Graph (DFG) therefrom.
class ControlFlowGraphVisitor : public Visitor {
 private:
  /// The nodes that were created most recently. Those are the parent nodes of the next node to be created.
  std::vector<GraphNode *> lastCreatedNodes;

  AccessType defaultAccessMode{AccessType::READ};

  std::set<std::pair<std::string, AccessType>> varAccess;

  /// The root node of the control flow graph.
  GraphNode *rootNode;

  bool handleExpressionsAsStatements{false};

 public:
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

  void visit(GetMatrixElement &elem) override;

  void visit(Ast &elem) override;

  void markVariableAccess(Variable &var);

  void markVariableAccess(const std::string &variableIdentifier, AccessType accessMode);

  GraphNode *appendStatementToGraph(AbstractNode &abstractStmt);

  GraphNode *appendStatementToCfg(AbstractNode &abstractStmt, const std::vector<GraphNode *> &parentNodes);

  [[nodiscard]] GraphNode *getRootNode() const;

  void postActionsStatementVisited(GraphNode *gNode);
};

#endif //AST_OPTIMIZER_INCLUDE_VISITOR_CONTROLFLOWGRAPHVISITOR_H_
