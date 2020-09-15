#include "ast_opt/ast/AbstractNode.h"
#include "ast_opt/ast/AbstractStatement.h"
#include "ast_opt/ast/AbstractExpression.h"
#include "ast_opt/ast/AbstractTarget.h"
#include "ast_opt/ast/BinaryExpression.h"
#include "ast_opt/ast/Block.h"
#include "ast_opt/ast/ExpressionList.h"
#include "ast_opt/ast/For.h"
#include "ast_opt/ast/Function.h"
#include "ast_opt/ast/FunctionParameter.h"
#include "ast_opt/ast/If.h"
#include "ast_opt/ast/IndexAccess.h"
#include "ast_opt/ast/Literal.h"
#include "ast_opt/ast/Return.h"
#include "ast_opt/ast/UnaryExpression.h"
#include "ast_opt/ast/Variable.h"
#include "ast_opt/ast/VariableAssignment.h"
#include "ast_opt/ast/VariableDeclaration.h"
#include "ast_opt/parser/Errors.h"
#include "ast_opt/parser/Parser.h"

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

  bool isSecret = false;
  if (it->hasValue(stork::reservedTokens::secret)) {
    isSecret = true;
    parseTokenValue(it, stork::reservedTokens::secret);
  }

  Datatype *datatype;
  switch (it->get_reserved_token()) {
    case stork::reservedTokens::kw_bool:
      datatype = new Datatype(Type::BOOL, isSecret);
      break;
    case stork::reservedTokens::kw_char:
      datatype = new Datatype(Type::CHAR, isSecret);
      break;
    case stork::reservedTokens::kw_int:
      datatype = new Datatype(Type::INT, isSecret);
      break;
    case stork::reservedTokens::kw_float:
      datatype = new Datatype(Type::FLOAT, isSecret);
      break;
    case stork::reservedTokens::kw_double:
      datatype = new Datatype(Type::DOUBLE, isSecret);
      break;
    case stork::reservedTokens::kw_string:
      datatype = new Datatype(Type::STRING, isSecret);
      break;
    case stork::reservedTokens::kw_void:
      datatype = new Datatype(Type::VOID);
      break;
    default:
      throw stork::unexpectedSyntaxError(std::to_string(it->getValue()), it->getLineNumber(), it->getCharIndex());
  }

  ++it;

  return datatype;
}

std::string parseDeclarationName(stork::tokens_iterator &it) {
  if (!it->isIdentifier()) {
    throw stork::unexpectedSyntaxError(std::to_string(it->getValue()), it->getLineNumber(), it->getCharIndex());
  }
  std::string ret = it->getIdentifier().name;
  ++it;
  return ret;
}

Function *Parser::parseFunctionStatement(stork::tokens_iterator &it) {
  // consume 'public'
  parseTokenValue(it, stork::reservedTokens::kw_public);

  // parse return type
  auto datatype = parseDatatype(it);

  // parse function name
  auto functionName = parseDeclarationName(it);

  // TODO: parse function parameters
  std::vector<std::unique_ptr<FunctionParameter>> functionParams;

  // TODO: parse block/body statements
  parseTokenValue(it, stork::reservedTokens::open_curly);
  while (!it->hasValue(stork::reservedTokens::close_curly)) {
    parseStatement(it);
  }
  parseTokenValue(it, stork::reservedTokens::close_curly);


//  auto func = new Function(datatype, functionName, functionParams, std::unique_ptr<Block>(body));
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
