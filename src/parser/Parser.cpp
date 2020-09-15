#include <ast_opt/parser/Errors.h>
#include "ast_opt/parser/Parser.h"
#include "ast_opt/ast/Literal.h"
#include "ast_opt/ast/Return.h"

std::unique_ptr<AbstractNode> Parser::parse(std::string) {
  return std::make_unique<LiteralBool>(0);
}

AbstractStatement *Parser::parse_statement(stork::tokens_iterator &it) {
  if (it->is_reserved_token()) {
    switch (it->get_reserved_token()) {
      case stork::reserved_token::kw_for:return parse_for_statement(it);
      case stork::reserved_token::kw_if:return parse_if_statement(it);
      case stork::reserved_token::kw_return:return parse_return_statement(it);
      case stork::reserved_token::open_curly:return parse_block_statement(it);
      case stork::reserved_token::kw_public: return parse_function_statement(it);
      default:
        // it starts with a type?
        return parse_variable_declaration_statement(it);
    }
  } else {
    // it better start with an identifier and be an assignment:
    return parse_variable_assignment_statement(it);
  }
}

AbstractExpression *Parser::parse_expression(stork::tokens_iterator &it) {
  //TODO:
  return nullptr;
}

/// consume token "value" and throw error if something different
void Parser::parse_token_value(stork::tokens_iterator &it, const stork::token_value &value) {
  if (it->has_value(value)) {
    ++it;
    return;
  }
  throw stork::expected_syntax_error(std::to_string(value), it->get_line_number(), it->get_char_index());
}

Function *Parser::parse_function_statement(stork::tokens_iterator &it) {
  return nullptr;
}
For *Parser::parse_for_statement(stork::tokens_iterator &it) {
  return nullptr;
}
If *Parser::parse_if_statement(stork::tokens_iterator &it) {
  return nullptr;
}

Return *Parser::parse_return_statement(stork::tokens_iterator &it) {
  parse_token_value(it, stork::reserved_token::kw_return);

  // Is it a return; i.e. no return value?
  if (it->has_value(stork::reserved_token::semicolon)) {
    return new Return();
  } else {
    AbstractExpression *p = parse_expression(it);
    return new Return(std::unique_ptr<AbstractExpression>(p));
  }
}

Block *Parser::parse_block_statement(stork::tokens_iterator &it) {
  return nullptr;
}
VariableDeclaration *Parser::parse_variable_declaration_statement(stork::tokens_iterator &it) {
  return nullptr;
}
VariableAssignment *Parser::parse_variable_assignment_statement(stork::tokens_iterator &it) {
  return nullptr;
}
