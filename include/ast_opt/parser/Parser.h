#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_PARSER_PARSER_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_PARSER_PARSER_H_

#include <ast_opt/ast/If.h>
#include "ast_opt/ast/AbstractStatement.h"
#include "ast_opt/ast/AbstractExpression.h"
#include "ast_opt/ast/Function.h"
#include "ast_opt/ast/For.h"
#include "ast_opt/ast/Return.h"
#include "ast_opt/ast/VariableDeclaration.h"
#include "ast_opt/ast/VariableAssignment.h"
#include "ast_opt/ast/AbstractNode.h"
#include "Tokenizer.h"

class Parser {
 public:
  static std::unique_ptr<AbstractNode> parse(std::string);

  AbstractStatement *parseStatement(stork::tokens_iterator &it);

  AbstractExpression *parseExpression(stork::tokens_iterator &it);

  Function *parseFunctionStatement(stork::tokens_iterator &it);

  For *parseForStatement(stork::tokens_iterator &it);

  Return *parseReturnStatement(stork::tokens_iterator &it);

  Block *parseBlockStatement(stork::tokens_iterator &it);

  VariableDeclaration *parseVariableDeclarationStatement(stork::tokens_iterator &it);

  VariableAssignment *parseVariableAssignmentStatement(stork::tokens_iterator &it);

  If *parseIfStatement(stork::tokens_iterator &it);

  void parseTokenValue(stork::tokens_iterator &it, const stork::token_value &value);
  Datatype *parseDatatype(stork::tokens_iterator &iterator);
};
#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_PARSER_PARSER_H_
