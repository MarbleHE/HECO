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
  /// A stack that keeps track of intermediate results. Each visit(..) of an expression (node that inherits from
  /// AbstractExpression) pushes its evaluation result on the stack so that the parent node can acccess the result.
  std::stack<std::unique_ptr<AbstractValue>> intermedResult;

  /// A map that provides datatype information for each scoped identifier.
  VariableDatatypeMap identifierDatatypes;

  /// A map that keeps track of ciphertexts that are created for declarations of secret variables.
  VariableMap<std::unique_ptr<AbstractCiphertext>> declaredCiphertexts;

  /// A map that keeps track of declared non-secret variables (i.e., cleartexts).
  VariableMap<std::unique_ptr<ICleartext>> declaredCleartexts;

  /// A reference to the factory to be used to create ciphertexts in the given input program AST. This is required as
  /// we need to create new ciphertexts for secret variable declarations.
  AbstractCiphertextFactory &factory;

  /// A map that contains the unique node ID for each secret tainted node, i.e., a node that must be secret because any
  /// of its dependant nodes (e.g., operand of binary expression) are secret.
  SecretTaintedNodesMap &secretTaintedMap;

  /// A map for noise budgets (only used if AbstractNoiseMeasuringCiphertexts are present)
  std::unordered_map<std::string, int> noise_map;

 public:
  /// Create a new SpecialRuntimeVisitor.
  /// \param factory A reference to the factory that should be used to create the ciphertexts.
  /// \param inputs The inputs of the program that should be executed with this SpecialRuntimeVisitor
  /// (by calling executeAst).
  /// \param secretTaintedNodesMap A map that contains the secret tainted nodes.
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

  /// A template method that is used to avoid code duplication in the visit methods for Literal<T>. As they do not have
  /// a common base type, we cannot create a single visit method for them.
  /// \tparam T The Literal type that is visited.
  /// \param elem (A reference) to the currently visiting Literal<T>.
  template<typename T>
  void visitHelper(Literal<T> &elem) {
    intermedResult.emplace(std::make_unique<Cleartext<T>>
                               (elem));
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

  /// Gets the top value of the stack containing intermediate results and removes it from the stack.
  /// \return (A std::unique_ptr of) the most recently pushed intermediate result.
  std::unique_ptr<AbstractValue> getNextStackElement();

  /// Prints the output of the given ast (outputAst) to the given stream (targetStream).
  /// \param outputAst The root node of the output AST, i.e., a single block with one or multiple assignments.
  /// \param targetStream The stream to write the textual representation of the AST to.
  [[maybe_unused]] void printOutput(AbstractNode &outputAst, std::ostream &targetStream = std::cout);

  /// Verifies that the structure of the input and output AST meet the expected form. More precisely,
  ///     * the input AST must consists of a single block with declaration statements,
  ///     * and the output AST of a single block with assignment statements.
  /// \tparam T The expected type of the statements in the block.
  /// \param astRootNode The root node of the AST to check.
  template<typename T>
  void checkAstStructure(AbstractNode &astRootNode);

  /// Gets the output specified by the given output AST.
  /// \param outputAst The root node of the output AST that describes the program's return values.
  /// \return (A copy of) a vector of pairs consisting of identifier (std::string) and the corresponding value
  /// (std::unique_ptr<AbstractValue>).
  OutputIdentifierValuePairs getOutput(AbstractNode &outputAst);

  /// Executes an input program given as AST. *NOTE*: As our RuntimeVisitor does not handle Functions yet, the visitor
  /// must be called on the function's Block. The missing function signature (input and outputs args) are derived from
  /// the input and output AST given in the constructor and the getOutput method, respectively.
  /// \param rootNode The root node of the input program.
  void executeAst(AbstractNode &rootNode);


  //TODO: Document Getter for noise map
  const std::unordered_map<std::string, int>& getNoiseMap();
};

#endif //GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_H_
