#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_COMPILER_COMPILER_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_COMPILER_COMPILER_H_

#include "ast_opt/ast/AbstractNode.h"
#include "ast_opt/runtime/RuntimeVisitor.h"

/// The compiler parses a program (given in JSON or pseudo-c++) and executes it.
class Compiler {
 private:
  /// Execute a program given as AST using the inputs provided as a block of variable declarations and returning
  /// OutputIdentifierValuePairs for the outputs specified as AST.
  /// \param programAst: program AST
  /// \param inputBlock: Block of statements, each declaring one input variable
  /// \param outputAst: output AST, must be a block of assignment statements, where on the RHS we only have
  ///                   variables or index access statements.
  /// \return: OutputIdentifierValuePairs for the identifiers specified in outputAst
  static OutputIdentifierValuePairs compileAst(std::unique_ptr<AbstractNode> programAst,
                                               std::unique_ptr<Block> inputBlock,
                                               std::unique_ptr<AbstractNode> outputAst);

  /// Transform a vector of identifiers to a block of (variable-to-variable) assignements.
  /// \param outputIdentifiers: vector of identifiers
  /// \return: an ABC AST block with a v = v; statement for every variable v in outputIdentifiers
  static std::unique_ptr<AbstractNode> buildOutputBlock(std::vector<std::string> outputIdentifiers);

 public:

  /// Execute a program given as C++ pseudo-code using the inputs provided as c++ pseudo-code and returning
  /// OutputIdentifierValuePairs for the outputs specified by identifiers given as vector.
  /// \param program: C++ pseudo-code program
  /// \param input: C++ pseudo-code input AST
  /// \param outputIdentifiers: vector of identifiers of the variables (in the input program) that should be returned.
  /// \return: OutputIdentifierValuePairs for the identifiers specified in outputIdentifiers
  static OutputIdentifierValuePairs compile(std::string program,
                                            std::string input,
                                            std::vector<std::string> outputIdentifiers);


  /// Execute a program given as JSON ABC AST using the inputs provided as JSON ABC AST and returning
  /// OutputIdentifierValuePairs for the outputs specified by identifiers given as vector.
  /// \param program: JSON ABC AST pseudo-code program
  /// \param input: JSON ABC AST pseudo-code input AST
  /// \param outputIdentifiers: vector of identifiers of the variables (in the input program) that should be returned.
  /// \return: OutputIdentifierValuePairs for the identifiers specified in outputIdentifiers
  static OutputIdentifierValuePairs compileJson(std::string program,
                                                std::string input,
                                                std::vector<std::string> outputIdentifiers);

  /// Execute a program given as ABC AST using the inputs provided as JSON ABC AST and returning
  /// OutputIdentifierValuePairs for the outputs specified by identifiers given as vector.
  /// \param program: ABC AST pseudo-code program
  /// \param input: JSON ABC AST pseudo-code input AST
  /// \param outputIdentifiers: vector of identifiers of the variables (in the input program) that should be returned.
  /// \return: OutputIdentifierValuePairs for the identifiers specified in outputIdentifiers
  static OutputIdentifierValuePairs compileJson(std::unique_ptr<AbstractNode> programAst,
                                                std::string input,
                                                std::vector<std::string> outputIdentifiers);
};
#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_COMPILER_COMPILER_H_
