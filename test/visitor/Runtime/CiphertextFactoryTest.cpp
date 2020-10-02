#include <include/ast_opt/visitor/Runtime/SealCiphertext.h>
#include "include/ast_opt/visitor/Runtime/SealCiphertextFactory.h"
#include "gtest/gtest.h"

TEST(CiphertextFactoryTest, createIntCiphertext) { /* NOLINT */
  std::unique_ptr<SealCiphertextFactory> scf = std::make_unique<SealCiphertextFactory>();
  std::vector<int64_t> data = {3, 3, 1, 4, 5, 9};
  std::unique_ptr<AbstractCiphertext> ctxt = scf->createCiphertext(data);

}
