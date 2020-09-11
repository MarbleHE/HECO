#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_PARSER_PARSER_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_PARSER_PARSER_H_

#include "memory"
#include "ast_opt/ast/AbstractNode.h"

class Parser{
 public:
  static std::unique_ptr<AbstractNode> parse(std::string);

};
#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_PARSER_PARSER_H_
