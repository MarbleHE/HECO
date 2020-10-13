#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_PARSER_PARSER_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_PARSER_PARSER_H_

#include <memory>
#include "ast_opt/utilities/Datatype.h"
#include "ast_opt/utilities/Operator.h"
#include "ast_opt/parser/Tokenizer.h"

// In order to avoid excessive compilation dependencies,
// we use forward-declarations rather than includes when possible
class AbstractExpression;
class AbstractNode;
class AbstractStatement;
class AbstractTarget;
class BinaryExpression;
class Block;
class ExpressionList;
class Function;
class FunctionParameter;
class For;
class If;
class IndexAccess;
class Operator;
class Return;
class UnaryExpression;
class Assignment;
class VariableDeclaration;
class Variable;

/// Vector to keep track of parsed nodes.
static std::vector<std::reference_wrapper<AbstractNode>> parsedNodes;

/// This datatype contains the datatype of the variable declaration currently processed, or nullptr if the recursion
/// currently is not parsing a variable declaration. It helps to distinguish between a literal and a bool.
static Datatype *varAssignmentDatatype;

/// The parser takes the
class Parser {
 private:

  static AbstractExpression *parseExpression(stork::tokens_iterator &it);

  static AbstractStatement *parseStatement(stork::tokens_iterator &it, bool gobbleTrailingSemicolon = true);

  static AbstractTarget *parseTarget(stork::tokens_iterator &it);

  static Block *parseBlockStatement(stork::tokens_iterator &it);

  static ExpressionList *parseExpressionList(stork::tokens_iterator &it);

  static For *parseForStatement(stork::tokens_iterator &it);

  static Function *parseFunctionStatement(stork::tokens_iterator &it);

  static FunctionParameter* parseFunctionParameter(stork::tokens_iterator &it);

  static If *parseIfStatement(stork::tokens_iterator &it);

  /// Returns a Literal of _some_ type without caring about type
  static AbstractExpression* parseLiteral(stork::tokens_iterator &it);

  static Return *parseReturnStatement(stork::tokens_iterator &it);

  static Variable *parseVariable(stork::tokens_iterator &it);

  static VariableDeclaration *parseVariableDeclarationStatement(stork::tokens_iterator &it);

  static Assignment *parseAssignmentStatement(stork::tokens_iterator &it);

  static void parseTokenValue(stork::tokens_iterator &it, const stork::token_value &value);

  static Datatype parseDatatype(stork::tokens_iterator &it);

  static Operator parseOperator(stork::tokens_iterator &it);

  static std::string parseIdentifier(stork::tokens_iterator &it);

  static Block *parseBlockOrSingleStatement(stork::tokens_iterator &it);

  static AbstractExpression *parseLiteral(stork::tokens_iterator &it, bool isNegative);

 public:
  /// Parses a given input program, returns (a unique ptr) to the created root node of the AST.
  /// \param s The program to parse given as string in a C++-like syntax.
  /// \return (A unique pointer) to the root node of the AST.
  static std::unique_ptr<AbstractNode> parse(std::string s);

  /// Parses a given input program, returns (a unique ptr) to the created root node of the AST and stores a reference to
  /// each created node (i.e., statement or expression) into the passed createdNodesList.
  /// \param s The program to parse given as string in a C++-like syntax.
  /// \param createdNodesList The list of parsed AbstractNodes.
  /// \return (A unique pointer) to the root node of the AST.
  static std::unique_ptr<AbstractNode> parse(std::string s,
                                             std::vector<std::reference_wrapper<AbstractNode>> &createdNodesList);
};
#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_PARSER_PARSER_H_
