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

  AbstractStatement *parse_statement(stork::tokens_iterator &it);

  AbstractExpression *parse_expression(stork::tokens_iterator &it);

  Function *parse_function_statement(stork::tokens_iterator &it);

  For *parse_for_statement(stork::tokens_iterator &it);

  Return *parse_return_statement(stork::tokens_iterator &it);

  Block *parse_block_statement(stork::tokens_iterator &it);

  VariableDeclaration *parse_variable_declaration_statement(stork::tokens_iterator &it);

  VariableAssignment *parse_variable_assignment_statement(stork::tokens_iterator &it);

  If *parse_if_statement(stork::tokens_iterator &it);

  void parse_token_value(stork::tokens_iterator &it, const stork::token_value &value);
};
#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_PARSER_PARSER_H_
