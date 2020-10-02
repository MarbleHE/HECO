#ifndef GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_CIPHERTEXTFACTORY_H_
#define GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_CIPHERTEXTFACTORY_H_

#include <initializer_list>
#include <string>

#include "ast_opt/visitor/Runtime/AbstractCiphertext.h"

class CiphertextFactory {
 public:
  virtual std::unique_ptr<AbstractCiphertext> createCiphertext(std::vector<int64_t> &data) = 0;
};

#endif //GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_CIPHERTEXTFACTORY_H_
