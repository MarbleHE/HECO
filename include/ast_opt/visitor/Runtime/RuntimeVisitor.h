#ifndef GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_H_
#define GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_H_

#include <stack>

#include "ast_opt/visitor/ScopedVisitor.h"
#include "ast_opt/visitor/Runtime/AbstractCiphertext.h"
#include "ast_opt/visitor/TypeCheckingVisitor.h"
#include "ast_opt/utilities/VariableMap.h"

// Forward declaration
class SpecialSecretBranchingVisitor;
class AbstractCiphertextFactory;

/// RuntimeVisitor uses the Visitor<T> template to allow specifying default behaviour
typedef Visitor<SpecialSecretBranchingVisitor> RuntimeVisitor;
typedef std::vector<std::pair<std::string, std::unique_ptr<AbstractCiphertext>>> OutputIdentifierValuePairs;

class SpecialRuntimeVisitor : public ScopedVisitor {
 private:
  std::stack<std::reference_wrapper<AbstractExpression>> intermedResult;

  /// this is produced by the TypeCheckingVisitor and lets us determine whether an identifier is secret
  // TODO: add this to the constructor
  VariableDatatypeMap identifierDatatypes;

  VariableMap<std::unique_ptr<AbstractCiphertext>> ciphertexts;

  // TODO: Think how to fix this as Literals do not have a common base class anymore
  std::unordered_map<ScopedIdentifier, std::reference_wrapper<AbstractExpression>> plainValues;

  AbstractCiphertextFactory &factory;

  SecretTaintedNodesMap &secretTaintedMap;

 public:
  SpecialRuntimeVisitor(AbstractCiphertextFactory &factory,
                        AbstractNode &inputs,
                        SecretTaintedNodesMap &secretTaintedNodesMap);

  void visit(BinaryExpression &elem) override;

  void visit(Block &elem) override;

  void visit(Call &elem) override;

  void visit(ExpressionList &elem) override;

  void visit(For &elem) override;

  void visit(Function &elem) override;

  void visit(If &elem) override;

  void visit(IndexAccess &elem) override;

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

  AbstractExpression &getNextStackElement();

  void printOutput(AbstractNode &outputAst, std::ostream &targetStream = std::cout);

  template<typename T>
  void checkAstStructure(AbstractNode &astRootNode);

  std::vector<std::pair<std::string, std::unique_ptr<AbstractCiphertext>>> getOutput(AbstractNode &outputAst);
};

#endif //GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_H_
