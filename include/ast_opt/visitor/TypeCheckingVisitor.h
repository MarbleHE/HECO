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

typedef std::unordered_map<ScopedIdentifier, Datatype> VariableDatatypeMap;

typedef std::unordered_map<std::string, bool> SecretTaintedNodesMap;

class SpecialTypeCheckingVisitor : public ScopedVisitor {
 private:
  /// Stores the number of nodes in typesVisitedNodes before the children of the currently visited nodes have been
  /// visited. For this, registerBeforeVisitChildren must be called before visiting the children.
  int numNodesBeforeVisitingChildren = 0;

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
  VariableDatatypeMap variablesDatatypeMap;

  /// Data types of all expression nodes in the program.
  std::unordered_map<std::string, Datatype> expressionsDatatypeMap;

  /// Stores for each node in the AST (identified by its unique node ID), if the node is tainted secret. This is the
  /// case if any of the operands in the node's expression are secret. The tainting stops at the first statement node
  /// encountered while walking the AST upwards. Consequently, if a statement in a block is tainted, the block will
  /// not be tainted. This implies that For statements are never tainted.
  SecretTaintedNodesMap secretTaintedNodes;

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

  /// Gets the list of secret tainted nodes, i.e., variables that are explicitly declared as being secret or that are
  /// involved in a computation that involves secret variables and as such also become secret.
  /// \return (A const reference) to the map containing (unique node ID, is tainted) pairs.
  [[nodiscard]] const SecretTaintedNodesMap &getSecretTaintedNodes() const;

  /// Gets the list of secret tainted nodes, i.e., variables that are explicitly declared as being secret or that are
  /// involved in a computation that involves secret variables and as such also become secret.
  /// \return (A reference) to the map containing (unique node ID, is tainted) pairs.
  SecretTaintedNodesMap &getSecretTaintedNodes();

  /// Stores a variable's datatype in the variable-datatype map.
  /// \param scopedIdentifier The variable's associated scoped identifier.
  /// \param datatype The datatype the given scoped identifier belongs to.
  void addVariableDatatype(ScopedIdentifier &scopedIdentifier, Datatype datatype);

  /// Remembers the number of nodes in typesVisitedNodes. After the children have been visited, calling
  /// discardChildrenDatatypes removes the datatype of the visited children from the stack. This mechanism is required
  /// for nodes that do not require children's type information or where the number of children is arbitrary and hence a
  /// simple pop() after visiting the children is not sufficient.
  void registerBeforeVisitChildren();

  /// Removes all datatypes from the typesVisitedNodes stack that were added in between now and the last call to
  /// registerBeforeVisitChildren. This is useful if you do not need type information of children nodes.
  void discardChildrenDatatypes();
};

#endif //GRAPHNODE_H_SRC_VISITOR_TYPECHECKINGVISITOR_H_
