#ifndef GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_H_
#define GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_H_

#include <stack>

#include "ast_opt/visitor/runtime/Cleartext.h"
#include "ast_opt/visitor/ScopedVisitor.h"
#include "ast_opt/visitor/runtime/AbstractCiphertext.h"
#include "ast_opt/visitor/TypeCheckingVisitor.h"
#include "ast_opt/utilities/VariableMap.h"

// Forward declaration
class SpecialRuntimeVisitor;
class AbstractCiphertextFactory;

/// A custom exception class to be used to break out of the RuntimeVisitor in case that we visited a Return statement.
/// This is important for programs that use Return statements to prematurely exit a program, e.g., in the body of a
/// For loop. The exception must be caught (and ignored) by the caller.
struct ReturnStatementReached : public std::exception {
  [[nodiscard]] const char *what() const noexcept override {
    return "Program reached Return statement. Exception raised to break out of RuntimeVisitor. "
           "This exception must be caught but can be ignored.";
  }
};

/// RuntimeVisitor uses the Visitor<T> template to allow specifying default behaviour
typedef Visitor<SpecialRuntimeVisitor> RuntimeVisitor;
typedef std::vector<std::pair<std::string, std::unique_ptr<AbstractValue>>> OutputIdentifierValuePairs;

class SpecialRuntimeVisitor : public ScopedVisitor {
 private:
  ///
  std::stack<std::unique_ptr<AbstractValue>> intermedResult;

  ///
  VariableDatatypeMap identifierDatatypes;

  ///
  VariableMap<std::unique_ptr<AbstractCiphertext>> declaredCiphertexts;

  VariableMap<std::unique_ptr<ICleartext>> declaredCleartexts;

  ///
  AbstractCiphertextFactory &factory;

  ///
  SecretTaintedNodesMap &secretTaintedMap;

 public:
  SpecialRuntimeVisitor(AbstractCiphertextFactory &factory,
                        AbstractNode &inputs,
                        SecretTaintedNodesMap &secretTaintedNodesMap);

  void visit(BinaryExpression &elem) override;

  void visit(Call &elem) override;

  void visit(ExpressionList &elem) override;

  void visit(For &elem) override;

  void visit(Function &elem) override;

  void visit(If &elem) override;

  void visit(IndexAccess &elem) override;

  template<typename T>
  void visitHelper(Literal<T> &elem) {
    intermedResult.emplace(std::make_unique<Cleartext<T>>(elem));
  }

  void visit(LiteralBool &elem) override;

  void visit(LiteralChar &elem) override;

  void visit(LiteralInt &elem) override;

  void visit(LiteralFloat &elem) override;

  void visit(LiteralDouble &elem) override;

  void visit(LiteralString &elem) override;

  void visit(OperatorExpression &elem) override;

  void visit(Return &elem) override;

  void visit(UnaryExpression &elem) override;

  void visit(Assignment &elem) override;

  void visit(VariableDeclaration &elem) override;

  void visit(Variable &elem) override;

  std::unique_ptr<AbstractValue> getNextStackElement();

  void printOutput(AbstractNode &outputAst, std::ostream &targetStream = std::cout);

  template<typename T>
  void checkAstStructure(AbstractNode &astRootNode);

  OutputIdentifierValuePairs getOutput(AbstractNode &outputAst);

  void executeAst(AbstractNode &rootNode);
};

#endif //GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_H_
