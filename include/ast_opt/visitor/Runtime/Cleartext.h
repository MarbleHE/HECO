#ifndef GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_CLEARTEXT_H_
#define GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_CLEARTEXT_H_

#include <vector>

#include "ast_opt/visitor/Runtime/AbstractValue.h"

class Cleartext : public AbstractValue {
 private:
  // the value this cleartext holds, currently only integers are supported as SEAL-BFV can only handle integers
  std::vector<int64_t> data;
 public:
  
};

#endif //GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_CLEARTEXT_H_
