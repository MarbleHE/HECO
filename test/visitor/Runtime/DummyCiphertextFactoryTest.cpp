#include <algorithm>

#include "gtest/gtest.h"

#include "include/ast_opt/visitor/runtime/DummyCiphertext.h"
#include "include/ast_opt/ast/ExpressionList.h"
#include "include/ast_opt/visitor/runtime/DummyCiphertextFactory.h"
#include "include/ast_opt/visitor/runtime/Cleartext.h"

#ifdef HAVE_SEAL_BFV

class SimulatorCiphertextFactoryTest : public ::testing::Test {
 protected:
  const int numCiphertextSlots = 4096;

  std::unique_ptr<DummyCiphertextFactory> scf;

  void SetUp() override {
   // scf = std::make_unique<DummyCiphertextFactory>(numCiphertextSlots);
  }




};


#endif