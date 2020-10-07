#ifndef GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_ABSTRACTPLAINTEXT_H_
#define GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_ABSTRACTPLAINTEXT_H_

#include "ast_opt/visitor/Runtime/AbstractValue.h"

class AbstractPlaintext : public AbstractValue {
 protected:
  // make sure that class is abstract, i.e., cannot be instantiated
  AbstractPlaintext() = default;

 public:
  /// Default destructor.
  ~AbstractPlaintext() override = default;
};

#endif //GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_ABSTRACTPLAINTEXT_H_
