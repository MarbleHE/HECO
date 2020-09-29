#ifndef GRAPHNODE_H_SRC_VISITOR_TYPECHECKINGVISITOR_H_
#define GRAPHNODE_H_SRC_VISITOR_TYPECHECKINGVISITOR_H_

#include "ast_opt/utilities/Datatype.h"
#include "ast_opt/visitor/ScopedVisitor.h"
#include <stack>

// Forward declaration
class SpecialTypeCheckingVisitor;
class AbstractExpression;

/// ControlFlowGraphVisitor uses the Visitor<T> template to allow specifying default behaviour
typedef Visitor<SpecialTypeCheckingVisitor> TypeCheckingVisitor;

class SpecialTypeCheckingVisitor : public ScopedVisitor {
 private:
  /// A temporary structure to keep track of data types visited in children of a statement.
  /// This stack should be empty after leaving a statement, otherwise it indicates that the statement did not properly
  /// clean up the stack.
  std::stack<Datatype> typesVisitedNodes;

  /// A vector to keep track of the data types (pair.first) that the expressions of the return statements have. It
  /// tracks whether the returned expression is a literal (pair.second) because in that case we do not need to check
  /// if the secretness specified in the function's signature is the same as the literal's type because we do not
  /// support defining secret constants.
  std::vector<std::pair<Datatype, bool>> returnExpressionTypes;

  /// Data types of the variables. This is derived from the variable's declaration.
  std::unordered_map<ScopedIdentifier,
                     Datatype> variablesDatatypeMap;

  /// Data types of all expression nodes in the program.
  std::unordered_map<std::string, Datatype> expressionsDatatypeMap;

  /// Stores for each node in the AST (identified by its unique node ID), if the node is tainted secret. This is the
  /// case if any of the operands in the node's expression are secret.
  std::unordered_map<std::string, bool> secretTaintedNodes;

  /// Internal function to check whether stack was cleaned up properly before leaving statement.
  void postStatementAction();

 public:
  void visit(BinaryExpression &elem) override;

  void visit(Call &elem) override;

  void visit(ExpressionList &elem) override;

  void visit(FunctionParameter &elem) override;

  void visit(IndexAccess &elem) override;

  void visit(LiteralBool &elem) override;

  void visit(LiteralChar &elem) override;

  void visit(LiteralInt &elem) override;

  void visit(LiteralFloat &elem) override;

  void visit(LiteralDouble &elem) override;

  void visit(LiteralString &elem) override;

  void visit(UnaryExpression &elem) override;

  void visit(VariableDeclaration &elem) override;

  void visit(Variable &elem) override;

  void visit(Block &elem) override;

  void visit(For &elem) override;

  void visit(Function &elem) override;

  void visit(If &elem) override;

  void visit(Return &elem) override;

  void visit(Assignment &elem) override;

  /// Returns the datatype of a given abstract expression.
  /// \param scopedIdentifier The scoped identifier for which the datatype should be determined.
  /// \return (A copy of) the datatype associated with the given scoped identifier.
  Datatype getVariableDatatype(ScopedIdentifier &scopedIdentifier);

  /// Gets the datatype of a given expression.
  /// \param expression The expression for which the datatype should be determined.
  /// \return (A copy of) the datatype associated with the given abstract expression node.
  Datatype getExpressionDatatype(AbstractExpression &expression);

  /// Checks whether a given node is secret tainted..
  /// \param uniqueNodeId The unique node ID of the node to be checked.
  /// \return (A bool) indicating whether the given node is secret tainted or not.
  bool isSecretTaintedNode(const std::string &uniqueNodeId);

  /// Checks whether both given data types are compatible to be used by the operands of an arithemtic expression.
  /// \param first The first datatype.
  /// \param second The second datatype.
  /// \return True if both are compatible to be used for operands of an arithmetic expression.
  static bool areCompatibleDatatypes(Datatype &first, Datatype &second);
};

#endif //GRAPHNODE_H_SRC_VISITOR_TYPECHECKINGVISITOR_H_
