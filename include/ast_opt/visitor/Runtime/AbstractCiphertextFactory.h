#ifndef GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_ABSTRACTCIPHERTEXTFACTORY_H_
#define GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_ABSTRACTCIPHERTEXTFACTORY_H_

#include <initializer_list>
#include <string>

class AbstractCiphertext;

class AbstractCiphertextFactory {
 public:
  virtual std::unique_ptr<AbstractCiphertext> createCiphertext(std::vector<int64_t> &data) = 0;

  virtual void decryptCiphertext(AbstractCiphertext &abstractCiphertext, std::vector<int64_t> &ciphertextData) = 0;
};

#endif //GRAPHNODE_H_INCLUDE_AST_OPT_VISITOR_RUNTIME_ABSTRACTCIPHERTEXTFACTORY_H_
