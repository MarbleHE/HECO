#include "include/ast_opt/parser/Parser.h"
#include "include/ast_opt/runtime/RuntimeVisitor.h"
#include "include/ast_opt/runtime/SimulatorCiphertextFactory.h"
#include "ast_opt/visitor/InsertModSwitchVisitor.h"
#include <ast_opt/visitor/GetAllNodesVisitor.h>
#include "ast_opt/visitor/ProgramPrintVisitor.h"
#include "ast_opt/visitor/PrintVisitor.h"
#include "ast_opt/visitor/NoisePrintVisitor.h"
#include "ast_opt/utilities/PerformanceSeal.h"
#include "ast_opt/runtime/SealCiphertextFactory.h"

#include "gtest/gtest.h"
#ifdef HAVE_SEAL_BFV


class BenchMarkEPFLModSwitch_bar : public ::testing::Test {

 protected:
  const int numCiphertextSlots = 16384;

  std::unique_ptr<SealCiphertextFactory> scf;
  std::unique_ptr<TypeCheckingVisitor> tcv;

  void SetUp() override {
    scf = std::make_unique<SealCiphertextFactory>(numCiphertextSlots);
    tcv = std::make_unique<TypeCheckingVisitor>();
  }

  void registerInputVariable(Scope &rootScope, const std::string &identifier, Datatype datatype) {
    auto scopedIdentifier = std::make_unique<ScopedIdentifier>(rootScope, identifier);
    rootScope.addIdentifier(identifier);
    tcv->addVariableDatatype(*scopedIdentifier, datatype);
  }
};

/// bar.v

TEST_F(BenchMarkEPFLModSwitch_bar, BarNoModSwitch) {

/// This test runs the bar circuit from the EPFL circuit collection WITHOUT any modswitch ops inserted using SEAL

// program's input
  const char *inputs = R""""(
secret int a0 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a1 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a2 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a3 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a4 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a5 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a6 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a7 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a8 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a9 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a10 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a11 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a12 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a13 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a14 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a15 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a16 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a17 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a18 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a19 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a20 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a21 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a22 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a23 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a24 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a25 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a26 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a27 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a28 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a29 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a30 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a31 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a32 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a33 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a34 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a35 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a36 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a37 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a38 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a39 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a40 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a41 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a42 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a43 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a44 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a45 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a46 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a47 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a48 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a49 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a50 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a51 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a52 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a53 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a54 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a55 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a56 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a57 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a58 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a59 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a60 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a61 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a62 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a63 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a64 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a65 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a66 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a67 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a68 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a69 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a70 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a71 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a72 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a73 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a74 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a75 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a76 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a77 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a78 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a79 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a80 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a81 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a82 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a83 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a84 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a85 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a86 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a87 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a88 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a89 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a90 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a91 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a92 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a93 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a94 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a95 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a96 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a97 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a98 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a99 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a100 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a101 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a102 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a103 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a104 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a105 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a106 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a107 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a108 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a109 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a110 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a111 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a112 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a113 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a114 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a115 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a116 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a117 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a118 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a119 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a120 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a121 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a122 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a123 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a124 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a125 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a126 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a127 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int shift0 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int shift1 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int shift2 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int shift3 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int shift4 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int shift5 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int shift6 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int one = {1,  1,   1,   1,  1, 1, 1,  1, 1, 1};
    )"""";
  auto astInput = Parser::parse(std::string(inputs));

// program specification
  const char *program = R""""(
secret int n264 = a77 *** shift0;
secret int n265 = shift1 *** n264;
secret int n266 = a78 *** (one --- shift0);
secret int n267 = shift1 *** n266;
secret int n268 = (one --- n265) *** (one --- n267);
secret int n269 = a80 *** (one --- shift0);
secret int n270 = (one --- shift1) *** n269;
secret int n271 = a79 *** shift0;
secret int n272 = (one --- shift1) *** n271;
secret int n273 = (one --- n270) *** (one --- n272);
secret int n274 = n268 *** n273;
secret int n275 = (one --- shift2) *** (one --- shift3);
secret int n276 = (one --- n274) *** n275;
secret int n277 = a73 *** shift0;
secret int n278 = shift1 *** n277;
secret int n279 = a74 *** (one --- shift0);
secret int n280 = shift1 *** n279;
secret int n281 = (one --- n278) *** (one --- n280);
secret int n282 = a76 *** (one --- shift0);
secret int n283 = (one --- shift1) *** n282;
secret int n284 = a75 *** shift0;
secret int n285 = (one --- shift1) *** n284;
secret int n286 = (one --- n283) *** (one --- n285);
secret int n287 = n281 *** n286;
secret int n288 = shift2 *** (one --- shift3);
secret int n289 = (one --- n287) *** n288;
secret int n290 = (one --- n276) *** (one --- n289);
secret int n291 = a65 *** shift0;
secret int n292 = shift1 *** n291;
secret int n293 = a66 *** (one --- shift0);
secret int n294 = shift1 *** n293;
secret int n295 = (one --- n292) *** (one --- n294);
secret int n296 = a68 *** (one --- shift0);
secret int n297 = (one --- shift1) *** n296;
secret int n298 = a67 *** shift0;
secret int n299 = (one --- shift1) *** n298;
secret int n300 = (one --- n297) *** (one --- n299);
secret int n301 = n295 *** n300;
secret int n302 = shift2 *** shift3;
secret int n303 = (one --- n301) *** n302;
secret int n304 = a69 *** shift0;
secret int n305 = shift1 *** n304;
secret int n306 = a70 *** (one --- shift0);
secret int n307 = shift1 *** n306;
secret int n308 = (one --- n305) *** (one --- n307);
secret int n309 = a72 *** (one --- shift0);
secret int n310 = (one --- shift1) *** n309;
secret int n311 = a71 *** shift0;
secret int n312 = (one --- shift1) *** n311;
secret int n313 = (one --- n310) *** (one --- n312);
secret int n314 = n308 *** n313;
secret int n315 = (one --- shift2) *** shift3;
secret int n316 = (one --- n314) *** n315;
secret int n317 = (one --- n303) *** (one --- n316);
secret int n318 = n290 *** n317;
secret int n319 = shift4 *** shift5;
    )"""";
  auto astProgram = Parser::parse(std::string(program));

// program's output
  const char *outputs = R""""(
      y = n319;
    )"""";

  auto astOutput = Parser::parse(std::string(outputs));
// create and prepopulate TypeCheckingVisitor
  auto rootScope = std::make_unique<Scope>(*astProgram);

  registerInputVariable(*rootScope, "a0", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a1", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a2", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a3", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a4", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a5", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a6", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a7", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a8", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a9", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a10", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a11", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a12", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a13", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a14", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a15", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a16", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a17", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a18", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a19", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a20", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a21", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a22", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a23", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a24", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a25", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a26", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a27", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a28", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a29", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a30", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a31", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a32", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a33", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a34", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a35", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a36", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a37", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a38", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a39", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a40", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a41", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a42", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a43", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a44", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a45", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a46", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a47", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a48", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a49", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a50", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a51", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a52", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a53", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a54", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a55", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a56", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a57", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a58", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a59", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a60", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a61", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a62", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a63", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a64", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a65", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a66", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a67", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a68", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a69", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a70", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a71", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a72", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a73", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a74", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a75", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a76", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a77", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a78", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a79", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a80", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a81", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a82", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a83", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a84", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a85", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a86", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a87", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a88", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a89", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a90", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a91", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a92", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a93", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a94", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a95", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a96", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a97", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a98", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a99", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a100", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a101", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a102", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a103", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a104", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a105", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a106", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a107", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a108", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a109", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a110", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a111", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a112", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a113", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a114", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a115", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a116", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a117", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a118", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a119", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a120", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a121", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a122", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a123", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a124", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a125", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a126", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a127", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "shift0", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "shift1", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "shift2", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "shift3", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "shift4", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "shift5", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "shift6", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "one", Datatype(Type::INT, true));

// run the program and get its output
  auto map = tcv->getSecretTaintedNodes();
  RuntimeVisitor srv(*scf, *astInput, map);


  std::chrono::high_resolution_clock::time_point time_start, time_end;
  std::chrono::microseconds time_diff;
  std::chrono::microseconds time_sum(0);
  std::vector<std::chrono::microseconds> time_vec;
  int count = 100;

  for (int i = 0; i < count; i++) {

    time_start = std::chrono::high_resolution_clock::now();

    srv.executeAst(*astProgram);

    time_end = std::chrono::high_resolution_clock::now();

    time_diff = std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start);
    time_vec.push_back(time_diff);
    time_sum += std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start);

    //std::cout << "Elapsed Time " << time_diff.count() << std::endl;
  }

  long long avg_time = std::chrono::duration_cast<std::chrono::microseconds>(time_sum).count()/(count);
  std::cout << count << std::endl;

  //calc std deviation
  long long standardDeviation = 0;
  for( int i = 0; i < time_vec.size(); ++i) {
    standardDeviation += (time_vec[i].count() - avg_time) * (time_vec[i].count() - avg_time);
  }

  std::cout << "Average evaluation time [" << avg_time << " microseconds]"
            << std::endl;
  std::cout << "Standard error: " << sqrt(double(standardDeviation) / time_vec.size())  / sqrt(time_vec.size())<< std::endl;

  // write to file
  std::cout << numCiphertextSlots << " , " << "bar : NO MODSWITCH" << std::endl;
  for (int i=0; i < time_vec.size(); i++) {
    std::cout << " , " << time_vec[i].count() << "\n";
  }

}

TEST_F(BenchMarkEPFLModSwitch_bar, BarModSwitch) {

/// This test runs the bar circuit from the EPFL circuit collection WITH modswitch ops inserted using SEAL

// program's input
  const char *inputs = R""""(
secret int a0 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a1 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a2 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a3 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a4 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a5 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a6 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a7 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a8 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a9 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a10 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a11 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a12 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a13 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a14 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a15 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a16 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a17 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a18 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a19 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a20 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a21 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a22 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a23 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a24 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a25 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a26 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a27 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a28 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a29 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a30 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a31 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a32 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a33 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a34 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a35 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a36 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a37 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a38 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a39 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a40 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a41 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a42 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a43 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a44 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a45 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a46 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a47 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a48 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a49 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a50 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a51 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a52 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a53 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a54 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a55 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a56 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a57 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a58 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a59 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a60 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a61 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a62 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a63 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a64 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a65 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a66 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a67 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a68 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a69 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a70 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a71 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a72 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a73 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a74 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a75 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a76 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a77 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a78 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a79 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a80 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a81 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a82 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a83 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a84 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a85 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a86 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a87 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a88 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a89 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a90 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a91 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a92 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a93 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a94 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a95 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a96 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a97 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a98 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a99 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a100 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a101 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a102 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a103 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a104 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a105 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a106 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a107 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a108 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a109 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a110 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a111 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a112 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a113 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a114 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a115 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a116 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a117 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a118 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a119 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a120 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a121 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a122 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a123 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a124 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a125 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a126 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int a127 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int shift0 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int shift1 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int shift2 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int shift3 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int shift4 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int shift5 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int shift6 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int one = {1,  1,   1,   1,  1, 1, 1,  1, 1, 1};
    )"""";
  auto astInput = Parser::parse(std::string(inputs));

// program specification
  const char *program = R""""(
    secret int n264 = (a77 *** shift0);
  secret int n265 = (shift1 *** n264);
  secret int n266 = (a78 *** (one --- shift0));
  secret int n267 = (shift1 *** n266);
  secret int n268 = (modswitch((one --- n265), 1) *** modswitch((one --- n267), 1));
  secret int n269 = (a80 *** (one --- shift0));
  secret int n270 = ((one --- shift1) *** n269);
  secret int n271 = (a79 *** shift0);
  secret int n272 = ((one --- shift1) *** n271);
  secret int n273 = ((one --- n270) *** (one --- n272));
  secret int n274 = (n268 *** modswitch(n273, 1));
  secret int n275 = ((one --- shift2) *** (one --- shift3));
  secret int n276 = ((modswitch(one, 1) --- n274) *** modswitch(n275, 1));
  secret int n277 = (a73 *** shift0);
  secret int n278 = (shift1 *** n277);
  secret int n279 = (a74 *** (one --- shift0));
  secret int n280 = (shift1 *** n279);
  secret int n281 = ((one --- n278) *** (one --- n280));
  secret int n282 = (a76 *** (one --- shift0));
  secret int n283 = ((one --- shift1) *** n282);
  secret int n284 = (a75 *** shift0);
  secret int n285 = ((one --- shift1) *** n284);
  secret int n286 = ((one --- n283) *** (one --- n285));
  secret int n287 = (n281 *** n286);
  secret int n288 = (shift2 *** (one --- shift3));
  secret int n289 = ((one --- n287) *** n288);
  secret int n290 = ((modswitch(one, 1) --- n276) *** modswitch((one --- n289), 1));
  secret int n291 = (a65 *** shift0);
  secret int n292 = (shift1 *** n291);
  secret int n293 = (a66 *** (one --- shift0));
  secret int n294 = (shift1 *** n293);
  secret int n295 = ((one --- n292) *** (one --- n294));
  secret int n296 = (a68 *** (one --- shift0));
  secret int n297 = ((one --- shift1) *** n296);
  secret int n298 = (a67 *** shift0);
  secret int n299 = ((one --- shift1) *** n298);
  secret int n300 = ((one --- n297) *** (one --- n299));
  secret int n301 = (n295 *** n300);
  secret int n302 = (shift2 *** shift3);
  secret int n303 = ((one --- n301) *** n302);
  secret int n304 = (a69 *** shift0);
  secret int n305 = (shift1 *** n304);
  secret int n306 = (a70 *** (one --- shift0));
  secret int n307 = (shift1 *** n306);
  secret int n308 = ((one --- n305) *** (one --- n307));
  secret int n309 = (a72 *** (one --- shift0));
  secret int n310 = ((one --- shift1) *** n309);
  secret int n311 = (a71 *** shift0);
  secret int n312 = ((one --- shift1) *** n311);
  secret int n313 = ((one --- n310) *** (one --- n312));
  secret int n314 = (n308 *** n313);
  secret int n315 = ((one --- shift2) *** shift3);
  secret int n316 = ((one --- n314) *** n315);
  secret int n317 = ((one --- n303) *** (one --- n316));
  secret int n318 = (n290 *** modswitch(n317, 1));
  secret int n319 = (shift4 *** shift5);
)"""";
  auto astProgram = Parser::parse(std::string(program));

// program's output
  const char *outputs = R""""(
      y = n319;
    )"""";

  auto astOutput = Parser::parse(std::string(outputs));
// create and prepopulate TypeCheckingVisitor
  auto rootScope = std::make_unique<Scope>(*astProgram);

  registerInputVariable(*rootScope, "a0", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a1", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a2", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a3", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a4", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a5", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a6", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a7", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a8", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a9", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a10", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a11", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a12", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a13", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a14", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a15", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a16", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a17", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a18", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a19", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a20", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a21", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a22", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a23", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a24", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a25", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a26", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a27", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a28", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a29", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a30", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a31", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a32", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a33", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a34", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a35", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a36", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a37", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a38", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a39", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a40", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a41", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a42", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a43", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a44", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a45", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a46", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a47", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a48", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a49", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a50", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a51", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a52", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a53", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a54", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a55", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a56", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a57", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a58", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a59", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a60", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a61", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a62", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a63", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a64", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a65", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a66", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a67", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a68", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a69", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a70", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a71", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a72", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a73", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a74", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a75", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a76", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a77", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a78", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a79", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a80", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a81", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a82", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a83", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a84", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a85", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a86", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a87", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a88", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a89", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a90", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a91", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a92", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a93", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a94", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a95", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a96", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a97", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a98", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a99", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a100", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a101", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a102", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a103", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a104", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a105", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a106", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a107", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a108", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a109", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a110", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a111", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a112", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a113", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a114", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a115", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a116", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a117", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a118", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a119", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a120", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a121", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a122", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a123", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a124", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a125", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a126", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a127", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "shift0", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "shift1", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "shift2", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "shift3", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "shift4", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "shift5", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "shift6", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "one", Datatype(Type::INT, true));

// run the program and get its output
  auto map = tcv->getSecretTaintedNodes();
  RuntimeVisitor srv(*scf, *astInput, map);


  std::chrono::high_resolution_clock::time_point time_start, time_end;
  std::chrono::microseconds time_diff;
  std::chrono::microseconds time_sum(0);
  std::vector<std::chrono::microseconds> time_vec;
  int count = 100;

  for (int i = 0; i < count; i++) {
    time_start = std::chrono::high_resolution_clock::now();

    srv.executeAst(*astProgram);

    time_end = std::chrono::high_resolution_clock::now();

    time_diff = std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start);
    time_vec.push_back(time_diff);
    time_sum += std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start);

    //std::cout << "Elapsed Time " << time_diff.count() << std::endl;
  }

  long long avg_time = std::chrono::duration_cast<std::chrono::microseconds>(time_sum).count()/(count);


  //calc std deviation
  long long standardDeviation = 0;
  for( int i = 0; i < time_vec.size(); ++i) {
    standardDeviation += (time_vec[i].count() - avg_time) * (time_vec[i].count() - avg_time);
  }

  std::cout << "Average evaluation time [" << avg_time << " microseconds]"
            << std::endl;
  std::cout << "Standard error: " << sqrt(double(standardDeviation) / time_vec.size())  / sqrt(time_vec.size())<< std::endl;

  // write to file
  std::cout << numCiphertextSlots << " , " << "bar : MODSWITCH" << std::endl;
  for (int i=0; i < time_vec.size(); i++) {
    std::cout << " , " << time_vec[i].count() << "\n";
  }


}

#endif