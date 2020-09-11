#include "ast_opt/parser/Parser.h"
#include "ast_opt/ast/Literal.h"

std::unique_ptr<AbstractNode> Parser::parse(std::string) {
  return std::make_unique<LiteralBool>(0);
}

