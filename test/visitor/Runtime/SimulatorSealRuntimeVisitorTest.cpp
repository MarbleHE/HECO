#include "include/ast_opt/ast/AbstractNode.h"
#include "include/ast_opt/parser/Parser.h"
#include "include/ast_opt/ast/Literal.h"
#include "include/ast_opt/visitor/runtime/RuntimeVisitor.h"
#include "include/ast_opt/visitor/runtime/SealCiphertextFactory.h"
#include <ast_opt/visitor/runtime/SimulatorCiphertext.h>
#include "include/ast_opt/visitor/runtime/SimulatorCiphertextFactory.h"
#include "ast_opt/utilities/PlaintextNorm.h"
#include "gtest/gtest.h"

#ifdef HAVE_SEAL_BFV

class SimulatorSealRuntimeVisitorTest : public ::testing::Test {
 protected:
  std::unique_ptr<SealCiphertextFactory> scf;
  std::unique_ptr<SimulatorCiphertextFactory> simcf;
  std::unique_ptr<TypeCheckingVisitor> tcv;

  void SetUp() override {
    scf = std::make_unique<SealCiphertextFactory>(4096);
    simcf = std::make_unique<SimulatorCiphertextFactory>(4096);
    tcv = std::make_unique<TypeCheckingVisitor>();
  }

  void registerInputVariable(Scope &rootScope, const std::string &identifier, Datatype datatype) {
    auto scopedIdentifier = std::make_unique<ScopedIdentifier>(rootScope, identifier);
    rootScope.addIdentifier(identifier);
    tcv->addVariableDatatype(*scopedIdentifier, datatype);
  }


};

#endif