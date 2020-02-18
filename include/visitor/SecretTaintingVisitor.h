#ifndef AST_OPTIMIZER_INCLUDE_VISITOR_SECRETTAINTINGVISITOR_H_
#define AST_OPTIMIZER_INCLUDE_VISITOR_SECRETTAINTINGVISITOR_H_

#include <unordered_map>
#include <vector>
#include "Visitor.h"

/// The SecretTaintingVisitor traverses through a given AST and marks all Nodes as "tainted" that in some way deal
/// with an encrypted variable.
/// For example, if [a,b] are encrypted variables and "int result = a + 2;" is a statement, then the VarAssignm
/// object, the BinaryExpr object and both of the literals would be marked as tainted.
class SecretTaintingVisitor : public Visitor {
 protected:
  /// The list of tainted Nodes.
  std::vector<std::string> taintedNodes;

  /// The list of tainted variables.
  std::vector<std::string> taintedVariables;

 public:
  /// Returns the list of all tainted nodes in the AST.
  /// \return A list consisting of the node's name (first) and True if the node is tainted, otherwise False (second).
  [[nodiscard]] const std::vector<std::string> &getSecretTaintingList() const;

  void visit(BinaryExpr &elem) override;
  void visit(Block &elem) override;
  void visit(Call &elem) override;
  void visit(CallExternal &elem) override;
  void visit(Function &elem) override;
  void visit(FunctionParameter &elem) override;
  void visit(If &elem) override;
  void visit(LiteralBool &elem) override;
  void visit(LiteralInt &elem) override;
  void visit(LiteralString &elem) override;
  void visit(LiteralFloat &elem) override;
  void visit(LogicalExpr &elem) override;
  void visit(Operator &elem) override;
  void visit(Return &elem) override;
  void visit(UnaryExpr &elem) override;
  void visit(VarAssignm &elem) override;
  void visit(VarDecl &elem) override;
  void visit(Variable &elem) override;
  void visit(While &elem) override;
  void visit(Ast &elem) override;
};

#endif //AST_OPTIMIZER_INCLUDE_VISITOR_SECRETTAINTINGVISITOR_H_
