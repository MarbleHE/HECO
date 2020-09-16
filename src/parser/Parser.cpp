#include <stack>
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
  // TODO: Implement me!
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
        // it starts with a data type (e.g., int, float)
        return parseVariableDeclarationStatement(it);
    }
  } else {
    // it start with an identifier -> must be an assignment
    return parseVariableAssignmentStatement(it);
  }
}

bool isOperator(stork::tokens_iterator &it) {
  return
      it->isReservedToken() &&
          (
              it->hasValue(stork::reservedTokens::add) ||
                  it->hasValue(stork::reservedTokens::sub) ||
                  it->hasValue(stork::reservedTokens::concat) ||
                  it->hasValue(stork::reservedTokens::mul) ||
                  it->hasValue(stork::reservedTokens::div) ||
                  it->hasValue(stork::reservedTokens::idiv) ||
                  it->hasValue(stork::reservedTokens::mod) ||
                  it->hasValue(stork::reservedTokens::bitwise_not) ||
                  it->hasValue(stork::reservedTokens::bitwise_and) ||
                  it->hasValue(stork::reservedTokens::bitwise_or) ||
                  it->hasValue(stork::reservedTokens::bitwise_xor) ||
                  it->hasValue(stork::reservedTokens::shiftl) ||
                  it->hasValue(stork::reservedTokens::shiftr) ||
                  it->hasValue(stork::reservedTokens::logical_not) ||
                  it->hasValue(stork::reservedTokens::logical_and) ||
                  it->hasValue(stork::reservedTokens::logical_or) ||
                  it->hasValue(stork::reservedTokens::eq) ||
                  it->hasValue(stork::reservedTokens::ne) ||
                  it->hasValue(stork::reservedTokens::lt) ||
                  it->hasValue(stork::reservedTokens::gt) ||
                  it->hasValue(stork::reservedTokens::le) ||
                  it->hasValue(stork::reservedTokens::ge)
          );
}

bool isLiteral(stork::tokens_iterator &it) {
  return it->isBool() || it->isChar() || it->isFloat() || it->isDouble() || it->isInteger() || it->isString();
}

/// Compares the precedence of this operator against another operator.
/// \return -1 if op1 is of lower precedence, 1 if it's of greater
/// precedence, and 0 if they're of equal precedence. If two operators are of
/// equal precedence, right associativity and parenthetical groupings must be
/// used to determine precedence.
int comparePrecedence(const Operator &op1, const Operator &op2) {
  //TODO: IMPLEMENT
  return 0;
}

bool isRightAssociative(const Operator &op) {
  //TODO:IMPLEMENT
  return false;
}

AbstractExpression *Parser::parseExpression(stork::tokens_iterator &it) {

  // if it begins with an "{" it must be an expression list which cannot be part of a greater expression
  if (it->hasValue(stork::reservedTokens::open_curly)) {
    return parseExpressionList(it);
  }

  // Shunting-yard algorithm: Keep a stack of operands and check precedence when you see an operator
  std::stack<AbstractExpression *, std::vector<AbstractExpression *>> operands;
  std::stack<Operator, std::vector<Operator>> operator_stack;

  bool running = true;
  while (running) {
    if (isOperator(it)) {
      Operator op1 = parseOperator(it);
      Operator *op2 = operator_stack.empty() ? nullptr : &operator_stack.top();

      while (!operator_stack.empty() && op2!=nullptr) {
        op2 = &operator_stack.top();
        if ((!isRightAssociative(op1) && comparePrecedence(op1, *op2)==0) || comparePrecedence(op1, *op2) < 0) {
          operator_stack.pop(); // pop
          AbstractExpression *rhs = operands.top();
          operands.pop();
          AbstractExpression *lhs = operands.top();
          operands.pop();
          operands.push(new BinaryExpression(std::unique_ptr<AbstractExpression>(lhs),
                                             *op2,
                                             std::unique_ptr<AbstractExpression>(rhs)));
        } else {
          break;
        }
      }
      operator_stack.push(op1);
    } else if (isLiteral(it)) {
      operands.push(parseLiteral(it));
    } else if (it->isIdentifier()) {
      operands.push(parseVariable(it));
    } else if (it->hasValue(stork::reservedTokens::open_round)) {
      // If we see an (, we have nested expressions going on, so use recursion.
      parseTokenValue(it, stork::reservedTokens::open_round);
      operands.push(parseExpression(it));
      parseTokenValue(it, stork::reservedTokens::close_round);
    } else {
      running = false;
    }
  }

  //TODO: Check that stack has been resolved correctly, otherwise throw exception
  return operands.top();
}

AbstractTarget *Parser::parseTarget(stork::tokens_iterator &it) {
  //Any valid target must begin with a Variable as its "root"
  Variable *v = parseVariable(it);

  std::vector<AbstractExpression *> indices;
  // if the next token is a "[" we need to keep on parsing
  while (it->hasValue(stork::reservedTokens::open_square)) {
    parseTokenValue(it, stork::reservedTokens::open_square);
    indices.push_back(parseExpression(it));
    parseTokenValue(it, stork::reservedTokens::close_square);
  }

  if (indices.empty()) {
    return v;
  } else {
    auto cur = new IndexAccess(std::unique_ptr<AbstractTarget>(v), std::unique_ptr<AbstractExpression>(indices[0]));
    for (size_t i = 1; i < indices.size(); ++i) {
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

Operator Parser::parseOperator(stork::tokens_iterator &it) {
  //TODO:
  throw std::runtime_error("NOT IMPLEMENTED");
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
    case stork::reservedTokens::kw_bool:datatype = Datatype(Type::BOOL, isSecret);
      break;
    case stork::reservedTokens::kw_char:datatype = Datatype(Type::CHAR, isSecret);
      break;
    case stork::reservedTokens::kw_int:datatype = Datatype(Type::INT, isSecret);
      break;
    case stork::reservedTokens::kw_float:datatype = Datatype(Type::FLOAT, isSecret);
      break;
    case stork::reservedTokens::kw_double:datatype = Datatype(Type::DOUBLE, isSecret);
      break;
    case stork::reservedTokens::kw_string:datatype = Datatype(Type::STRING, isSecret);
      break;
    case stork::reservedTokens::kw_void:datatype = Datatype(Type::VOID);
      break;
    default:throw stork::unexpectedSyntaxError(std::to_string(it->getValue()), it->getLineNumber(), it->getCharIndex());
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

  // parse block/body statements
  auto block = std::unique_ptr<Block>(parseBlockStatement(it));

  return new Function(datatype, functionName, std::move(functionParams), std::move(block));
}

For *Parser::parseForStatement(stork::tokens_iterator &it) {
  // TODO: Implement me!
  return nullptr;
}

If *Parser::parseIfStatement(stork::tokens_iterator &it) {
  // TODO: Implement me!
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
  // parse block/body statements
  parseTokenValue(it, stork::reservedTokens::open_curly);
  std::vector<std::unique_ptr<AbstractStatement>> blockStatements;
  while (!it->hasValue(stork::reservedTokens::close_curly)) {
    blockStatements.push_back(std::unique_ptr<AbstractStatement>(parseStatement(it)));
  }
  parseTokenValue(it, stork::reservedTokens::close_curly);
  return new Block(std::move(blockStatements));
}

AbstractExpression* Parser::parseTargetValue(stork::tokens_iterator &it) {
  AbstractExpression* value;

  if (it->isBool()) {
    value = new LiteralBool(it->getBool());
  } else if (it->isChar()) {
    value = new LiteralChar(it->getChar());
  } else if (it->isFloat()) {
    value = new LiteralFloat(it->getFloat());
  } else if (it->isDouble()) {
    value = new LiteralDouble(it->getDouble());
  } else if (it->isString()) {
    value = new LiteralString(it->getString());
  } else if (it->isInteger()) {
    value = new LiteralInt(it->getInteger());
  } else {
    throw stork::unexpectedSyntaxError(std::to_string(it->getValue()), it->getLineNumber(), it->getCharIndex());
  }

  ++it;

  return value;
}

VariableDeclaration *Parser::parseVariableDeclarationStatement(stork::tokens_iterator &it) {
  // the variable's datatype
  auto datatype = parseDatatype(it);

  // the variable's name
  auto identifier = parseDeclarationName(it);
  auto variable = std::make_unique<Variable>(identifier);

  // the variable's assigned value, if any assigned
  if (!it->hasValue(stork::reservedTokens::semicolon)) {
    parseTokenValue(it, stork::reservedTokens::assign);
    AbstractExpression *value = parseTargetValue(it);
    // the trailing semicolon
    parseTokenValue(it, stork::reservedTokens::semicolon);
    return new VariableDeclaration(datatype, std::move(variable), std::unique_ptr<AbstractExpression>(value));
  } else {
    // the trailing semicolon
    parseTokenValue(it, stork::reservedTokens::semicolon);
    return new VariableDeclaration(datatype, std::move(variable));
  }
}

VariableAssignment *Parser::parseVariableAssignmentStatement(stork::tokens_iterator &it) {
  // the variable's name
  auto identifier = parseDeclarationName(it);
  auto variable = std::make_unique<Variable>(identifier);

  // the variable's assigned value
  parseTokenValue(it, stork::reservedTokens::assign);
  AbstractExpression *value = parseTargetValue(it);

  // the trailing semicolon
  parseTokenValue(it, stork::reservedTokens::semicolon);

  return new VariableAssignment(std::move(variable), std::unique_ptr<AbstractExpression>(value));
}
