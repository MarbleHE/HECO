#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_PARSER_FILE_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_PARSER_FILE_H_

#include <string>
#include "ast_opt/parser/Errors.h"

namespace stork {

class File {
 private:
  FILE *_fp;

 public:
  explicit File(const char *path);

  ~File();

  int operator()();

  File(const File &) = delete;

  void operator=(const File &) = delete;
};
}
#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_PARSER_FILE_H_
