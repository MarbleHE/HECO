#ifndef GRAPHNODE_H_SRC_VISITOR_TYPECHECKINGVISITOR_H_
#define GRAPHNODE_H_SRC_VISITOR_TYPECHECKINGVISITOR_H_

#include "ast_opt/utilities/Datatype.h"
#include "ast_opt/visitor/ScopedVisitor.h"
#include <stack>

// Forward declaration
class SpecialTypeCheckingVisitor;

/// ControlFlowGraphVisitor uses the Visitor<T> template to allow specifying default behaviour
typedef Visitor<SpecialTypeCheckingVisitor> TypeCheckingVisitor;

class SpecialTypeCheckingVisitor : public ScopedVisitor {
 private:
  /// a temporary structure to keep track of data types visited in children of a statement
  std::stack<Datatype> typesVisitedNodes;

  /// data types of the variables derived from their declaration
  std::unordered_map<ScopedIdentifier,
                     std::reference_wrapper<Datatype>,
                     ScopedIdentifierHashFunction> variablesDatatypeMap;

  /// data types of the expression nodes
  std::unordered_map<std::string, Datatype> expressionsDatatypeMap;

  /// stores for each node in the AST (identified by its unique node ID), if the node is tainted secret
  std::unordered_map<std::string, bool> secretTaintedNodes;

 public:
  void visit(BinaryExpression &elem) override;

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

  Datatype &getVariableDatatype(ScopedIdentifier &scopedIdentifier);

  void checkStatementFinished();
};

#endif //GRAPHNODE_H_SRC_VISITOR_TYPECHECKINGVISITOR_H_
