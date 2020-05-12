#ifndef AST_OPTIMIZER_SRC_MOCKUP_CLASSES_PLAINTEXT_H_
#define AST_OPTIMIZER_SRC_MOCKUP_CLASSES_PLAINTEXT_H_

#include "Ciphertext.h"

class Plaintext : public Ciphertext {
 public:
  explicit Plaintext(const Ciphertext &ctxt);

  explicit Plaintext(double scalar, int numCiphertextSlots = Ciphertext::DEFAULT_NUM_SLOTS);

  explicit Plaintext(const std::vector<double> &data, int numCiphertextSlots = Ciphertext::DEFAULT_NUM_SLOTS);
};

#endif //AST_OPTIMIZER_SRC_MOCKUP_CLASSES_PLAINTEXT_H_
