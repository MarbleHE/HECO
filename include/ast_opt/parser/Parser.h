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
class VariableAssignment;
class VariableDeclaration;
class Variable;

/// The parser takes the
class Parser {
 public:
  static std::unique_ptr<AbstractNode> parse(std::string);

  static AbstractExpression *parseExpression(stork::tokens_iterator &it);

  static AbstractStatement *parseStatement(stork::tokens_iterator &it);

  static AbstractTarget *parseTarget(stork::tokens_iterator &it);

  static BinaryExpression* parseBinaryExpression(stork::tokens_iterator &it);

  static Block *parseBlockStatement(stork::tokens_iterator &it);

  static ExpressionList* parseExpressionList(stork::tokens_iterator &it);

  static For *parseForStatement(stork::tokens_iterator &it);

  static Function *parseFunctionStatement(stork::tokens_iterator &it);

  static FunctionParameter* parseFunctionParameter(stork::tokens_iterator &it);

  static If *parseIfStatement(stork::tokens_iterator &it);

  /// Returns a Literal of _some_ type without caring about type
  static AbstractExpression* parseLiteral(stork::tokens_iterator &it);

  static Return *parseReturnStatement(stork::tokens_iterator &it);

  static UnaryExpression* parseUnaryExpression(stork::tokens_iterator &it);

  static Variable *parseVariable(stork::tokens_iterator &it);

  static VariableDeclaration *parseVariableDeclarationStatement(stork::tokens_iterator &it);

  static VariableAssignment *parseVariableAssignmentStatement(stork::tokens_iterator &it);

  static void parseTokenValue(stork::tokens_iterator &it, const stork::token_value &value);

  static Datatype parseDatatype(stork::tokens_iterator &it);

  static Operator parseOperator(stork::tokens_iterator &it);

  static AbstractExpression *parseTargetValue(stork::tokens_iterator &it);
};
#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_PARSER_PARSER_H_
