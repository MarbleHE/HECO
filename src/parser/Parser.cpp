#include <ast_opt/parser/Errors.h>
#include "ast_opt/parser/Parser.h"
#include "ast_opt/ast/Literal.h"
#include "ast_opt/ast/Return.h"

std::unique_ptr<AbstractNode> Parser::parse(std::string) {
  return std::make_unique<LiteralBool>(0);
}

AbstractStatement *Parser::parseStatement(stork::tokens_iterator &it) {
  if (it->isReservedToken()) {
    switch (it->get_reserved_token()) {
      case stork::reservedTokens::kw_for:return parseForStatement(it);
      case stork::reservedTokens::kw_if:return parseIfStatement(it);
      case stork::reservedTokens::kw_return:return parseReturnStatement(it);
      case stork::reservedTokens::open_curly:return parseBlockStatement(it);
      case stork::reservedTokens::kw_public: return parseFunctionStatement(it);
      default:
        // it starts with a data type? (e.g., int)
        return parseVariableDeclarationStatement(it);
    }
  } else {
    // it better start with an identifier and be an assignment:
    return parseVariableAssignmentStatement(it);
  }
}

AbstractExpression *Parser::parseExpression(stork::tokens_iterator &it) {
  //TODO:
  return nullptr;
}

/// consume token "value" and throw error if something different
void Parser::parseTokenValue(stork::tokens_iterator &it, const stork::token_value &value) {
  if (it->hasValue(value)) {
    ++it;
    return;
  }
  throw stork::throwExpectedSyntaxError(std::to_string(value), it->getLineNumber(), it->getCharIndex());
}

Function *Parser::parseFunctionStatement(stork::tokens_iterator &it) {
  return nullptr;
}
For *Parser::parseForStatement(stork::tokens_iterator &it) {
  return nullptr;
}
If *Parser::parseIfStatement(stork::tokens_iterator &it) {
  return nullptr;
}

Return *Parser::parseReturnStatement(stork::tokens_iterator &it) {
  parseTokenValue(it, stork::reservedTokens::kw_return);

  // Is it a return; i.e. no return value?
  if (it->hasValue(stork::reservedTokens::semicolon)) {
    return new Return();
  } else {
    AbstractExpression *p = parseExpression(it);
    return new Return(std::unique_ptr<AbstractExpression>(p));
  }
}

Block *Parser::parseBlockStatement(stork::tokens_iterator &it) {
  return nullptr;
}
VariableDeclaration *Parser::parseVariableDeclarationStatement(stork::tokens_iterator &it) {
  return nullptr;
}
VariableAssignment *Parser::parseVariableAssignmentStatement(stork::tokens_iterator &it) {
  return nullptr;
}
