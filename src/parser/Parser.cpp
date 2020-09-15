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

AbstractTarget *Parser::parseTarget(stork::tokens_iterator &it) {
  //Any valid target must begin with a Variable as its "root"
  Variable* v = parseVariable(it);

  std::vector<AbstractExpression*> indices;
  // if the next token is a "[" we need to keep on parsing
  while(it->hasValue(stork::reservedTokens::open_square)) {
    parseTokenValue(it, stork::reservedTokens::open_square);
    indices.push_back(parseExpression(it));
    parseTokenValue(it, stork::reservedTokens::close_square);
  }

  if(indices.empty()) {
    return v;
  } else {
    auto cur = new IndexAccess(std::unique_ptr<AbstractTarget>(v), std::unique_ptr<AbstractExpression>(indices[0]));
    for(size_t i = 1; i < indices.size(); ++i) {
      cur = new IndexAccess(std::unique_ptr<AbstractTarget>(cur), std::unique_ptr<AbstractExpression>(indices[i]));
    }
    return cur;
  }
}

BinaryExpression *Parser::parseBinaryExpression(stork::tokens_iterator &it) {
  //TODO:
  return nullptr;
}

ExpressionList *Parser::parseExpressionList(stork::tokens_iterator &it) {
  //TODO:
  return nullptr;
}

FunctionParameter *Parser::parseFunctionParameter(stork::tokens_iterator &it) {
  //TODO:
  return nullptr;
}

AbstractExpression *Parser::parseLiteral(stork::tokens_iterator &it) {
  //TODO:
  return nullptr;
}


UnaryExpression *Parser::parseUnaryExpression(stork::tokens_iterator &it) {
  //TODO:
  return nullptr;
}

Variable *Parser::parseVariable(stork::tokens_iterator &it) {
  //TODO:
  return nullptr;
}

Operator *Parser::parseOperator(stork::tokens_iterator &it) {
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

Datatype Parser::parseDatatype(stork::tokens_iterator &it) {
  if (!it->isReservedToken()) {
    throw stork::unexpectedSyntaxError(std::to_string(it->getValue()), it->getLineNumber(), it->getCharIndex());
  }

  bool isSecret = false;
  if (it->hasValue(stork::reservedTokens::secret)) {
    isSecret = true;
    parseTokenValue(it, stork::reservedTokens::secret);
  }

  // just a placeholder as value-less constructor does not exist
  Datatype datatype(Type::VOID);
  switch (it->get_reserved_token()) {
    case stork::reservedTokens::kw_bool:
      datatype = Datatype(Type::BOOL, isSecret);
      break;
    case stork::reservedTokens::kw_char:
      datatype = Datatype(Type::CHAR, isSecret);
      break;
    case stork::reservedTokens::kw_int:
      datatype = Datatype(Type::INT, isSecret);
      break;
    case stork::reservedTokens::kw_float:
      datatype = Datatype(Type::FLOAT, isSecret);
      break;
    case stork::reservedTokens::kw_double:
      datatype = Datatype(Type::DOUBLE, isSecret);
      break;
    case stork::reservedTokens::kw_string:
      datatype = Datatype(Type::STRING, isSecret);
      break;
    case stork::reservedTokens::kw_void:
      datatype = Datatype(Type::VOID);
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

FunctionParameter *Parser::parseFunctionParameter(stork::tokens_iterator &it) {
  auto datatype = parseDatatype(it);
  auto identifier = parseDeclarationName(it);

  auto functionParameter = new FunctionParameter(datatype, identifier);

  // consume comma that separates this parameter from the next one
  // the caller is responsible for calling this method again for parsing the next parameter
  if (it->hasValue(stork::reservedTokens::comma)) {
    parseTokenValue(it, stork::reservedTokens::comma);
  }

  return functionParameter;
}

Function *Parser::parseFunctionStatement(stork::tokens_iterator &it) {
  // consume 'public'
  parseTokenValue(it, stork::reservedTokens::kw_public);

  // parse return type
  auto datatype = parseDatatype(it);

  // parse function name
  auto functionName = parseDeclarationName(it);

  // parse function parameters
  parseTokenValue(it, stork::reservedTokens::open_round);
  std::vector<std::unique_ptr<FunctionParameter>> functionParams;
  while (!it->hasValue(stork::reservedTokens::close_round)) {
    functionParams.push_back(std::unique_ptr<FunctionParameter>(parseFunctionParameter(it)));
  }
  parseTokenValue(it, stork::reservedTokens::close_round);

  // TODO: parse block/body statements
  parseTokenValue(it, stork::reservedTokens::open_curly);
  std::vector<std::unique_ptr<AbstractStatement>> blockStatements;
  while (!it->hasValue(stork::reservedTokens::close_curly)) {
//    blockStatements.push_back(std::unique_ptr<AbstractStatement>(parseStatement(it)));
  }
  parseTokenValue(it, stork::reservedTokens::close_curly);
  auto block = new Block(std::move(blockStatements));

//  return new Function(datatype, functionName, functionParams, std::unique_ptr<Block>(block));
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
