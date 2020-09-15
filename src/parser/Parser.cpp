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
  throw stork::expectedSyntaxError(std::to_string(value), it->getLineNumber(), it->getCharIndex());
}

Datatype *Parser::parseDatatype(stork::tokens_iterator &it) {
  if (!it->isReservedToken()) {
    throw stork::unexpectedSyntaxError(std::to_string(it->getValue()), it->getLineNumber(), it->getCharIndex());
  }

  // TODO: Introduce secret counterparts and adapts datatype parsing appropriately
  switch (it->get_reserved_token()) {
    case stork::reservedTokens::kw_bool:
    case stork::reservedTokens::kw_secret_bool:
      return new Datatype(Type::BOOL, it->get_reserved_token() == stork::reservedTokens::kw_secret_bool);

    case stork::reservedTokens::kw_char:
    case stork::reservedTokens::kw_secret_char:
      return new Datatype(Type::CHAR, it->get_reserved_token() == stork::reservedTokens::kw_secret_char);


    case stork::reservedTokens::kw_int:
    case stork::reservedTokens::kw_secret_int:
      return new Datatype(Type::INT);

    case stork::reservedTokens::kw_float:
      return new Datatype(Type::FLOAT);
    case stork::reservedTokens::kw_string:
      return new Datatype(Type::STRING);

    case stork::reservedTokens::kw_void:
      return new Datatype(Type::VOID);
    default:
      throw stork::unexpectedSyntaxError(std::to_string(it->getValue()), it->getLineNumber(), it->getCharIndex());
  }

  //   kw_bool,
  //  kw_char,
  //  kw_int,
  //  kw_float,
  //  kw_double,
  //  kw_string,
  //  kw_void,
}


Function *Parser::parseFunctionStatement(stork::tokens_iterator &it) {
  // consume 'public'
  parseTokenValue(it, stork::reservedTokens::kw_public);

  // TODO: parse return type
  auto dt = parseDatatype(it);

  // TODO: parse function name

  // TODO: parse function parameters

  // TODO: parse block/body statements
  parseTokenValue(it, stork::reservedTokens::open_curly);
  while (!it->hasValue(stork::reservedTokens::close_curly)) {
    parseStatement(it);
  }
  parseTokenValue(it, stork::reservedTokens::close_curly);


//  auto func = new Function();
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
