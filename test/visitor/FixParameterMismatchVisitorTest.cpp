#include <gmp.h>
#include "include/ast_opt/parser/Parser.h"
#include "include/ast_opt/runtime/RuntimeVisitor.h"
#include "include/ast_opt/runtime/SimulatorCiphertextFactory.h"
#include "ast_opt/visitor/InsertModSwitchVisitor.h"
#include <ast_opt/visitor/GetAllNodesVisitor.h>
#include "../ASTComparison.h"
#include "ast_opt/visitor/ProgramPrintVisitor.h"
#include "ast_opt/visitor/PrintVisitor.h"
#include "ast_opt/visitor/NoisePrintVisitor.h"
#include "ast_opt/utilities/PerformanceSeal.h"
#include "ast_opt/visitor/FixParamMismatchVisitor.h"

#include "gtest/gtest.h"
#ifdef HAVE_SEAL_BFV

class FixParameterMismatchVisitorVisitorTest : public ::testing::Test {

 protected:
  std::unique_ptr<SimulatorCiphertextFactory> scf;
  std::unique_ptr<TypeCheckingVisitor> tcv;

  void SetUp() override {
    scf = std::make_unique<SimulatorCiphertextFactory>(16384);
    tcv = std::make_unique<TypeCheckingVisitor>();

    print_parameters(scf->getContext());
  }

  void registerInputVariable(Scope &rootScope, const std::string &identifier, Datatype datatype) {
    auto scopedIdentifier = std::make_unique<ScopedIdentifier>(rootScope, identifier);
    rootScope.addIdentifier(identifier);
    tcv->addVariableDatatype(*scopedIdentifier, datatype);
  }

  // calculates initial noise heuristic of a freshly encrypted ciphertext
  uint64_t calcInitNoiseHeuristic() {
    uint64_t plain_modulus = scf->getContext().first_context_data()->parms().plain_modulus().value();
    uint64_t poly_modulus = scf->getContext().first_context_data()->parms().poly_modulus_degree();
    mpz_t result_noise;
    mpz_init(result_noise);
    mpz_t plain_mod;
    mpz_init(plain_mod);
    mpz_init_set_ui(plain_mod, plain_modulus);
    mpz_t poly_mod;
    mpz_init(poly_mod);
    mpz_init_set_ui(poly_mod, poly_modulus);
    // summand_one = n * (t-1) / 2
    mpz_t summand_one;
    mpz_init(summand_one);
    mpz_sub_ui(summand_one, plain_mod, 1);
    mpz_mul(summand_one, summand_one, poly_mod);
    mpz_div_ui(summand_one, summand_one, 2);
    // summand_two = 2 * sigma * sqrt(12 * n ^2 + 9 * n)
    mpz_t summand_two;
    mpz_init(summand_two);
    mpz_pow_ui(summand_two, poly_mod, 2);
    mpz_mul_ui(summand_two, summand_two, 12);
    mpz_t poly_mod_times_nine;
    mpz_init(poly_mod_times_nine);
    mpz_mul_ui(poly_mod_times_nine, poly_mod, 9);
    mpz_add(summand_two, summand_two, poly_mod_times_nine);
    mpz_sqrt(summand_two, summand_two);
    mpz_mul_ui(summand_two, summand_two, long(6.4)); // sigma = 3.2
    mpz_t sum;
    // sum = summand_1 + summand_2
    mpz_init(sum);
    mpz_add(sum, summand_one, summand_two);
    // result_noise = t * sum
    mpz_mul(result_noise, sum, plain_mod);
    size_t
        coeff_modulus_significant_bit_count = scf->getContext().first_context_data()->total_coeff_modulus_bit_count();
    size_t log_noise = mpz_sizeinbase(result_noise, 2);
    return std::max(int(coeff_modulus_significant_bit_count - log_noise - 1), 0);
  }
};


TEST_F(FixParameterMismatchVisitorVisitorTest, insertOne) {

  /// ((x^4 + y) * z^4) * w
  /// expected: insertion of a single modswitch operatioon applied to operands (x^4 + y) and z^4
  /// and an additional modswitch operation to w as result of the FixParameterMismatchVisitor

  // program's input
  const char *inputs = R""""(
      secret int __input0__ = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
      secret int __input1__ = {24, 34, 222,   4,    1, 4,   9, 22, 1, 3};
      secret int __input2__ = {24, 34, 222,   4,    1, 4,   9, 22, 1, 3};
      secret int __input3__ = {24, 34, 222,   4,    1, 4,   9, 22, 1, 3};
    )"""";
  auto astInput = Parser::parse(std::string(inputs));

  // program specification
  const char *program = R""""(
      secret int powx2 = __input0__ *** __input0__;
      secret int powx3 = powx2 *** __input0__;
      secret int powx4 = powx3 *** __input0__;
      secret int powx4plusy = powx4 +++ __input1__;
      secret int powz2 = __input2__ *** __input2__;
      secret int powz3 = powz2 *** __input2__;
      secret int powz4 = powz3 *** __input2__;
      secret int result1 = powx4plusy *** powz4;
      secret int result = result1 *** __input3__;
      return result;
    )"""";
  auto astProgram = Parser::parse(std::string(program));

  // program's output
  const char *outputs = R""""(
      y = result;
    )"""";
  auto astOutput = Parser::parse(std::string(outputs));

  // create and prepopulate TypeCheckingVisitor
  auto rootScope = std::make_unique<Scope>(*astProgram);
  registerInputVariable(*rootScope, "__input0__", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "__input1__", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "__input2__", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "__input3__", Datatype(Type::INT, true));

  tcv->setRootScope(std::move(rootScope));
  astProgram->accept(*tcv);

  std::stringstream ss;
  ProgramPrintVisitor p(ss);
  astProgram->accept(p);
  std::cout << ss.str() << std::endl;

  // run the program and get its output (using the SimulatorCiphertextVisitor)
  auto map = tcv->getSecretTaintedNodes();
  RuntimeVisitor srv(*scf, *astInput, map);
  srv.executeAst(*astProgram);

  GetAllNodesVisitor vis;
  astProgram->accept(vis);

  //  map of coeff modulus vectors: we initially populate this map with the original coeff_modulus vector for each node in the AST
  auto coeff_modulus = scf->getContext().first_context_data()->parms().coeff_modulus();

  // initially every node has the same ctxtmodulus vector: this map should change at the binary expreesion after inserting
  // the modswitch
  std::unordered_map<std::string, std::vector<seal::Modulus>> coeffmodulusmap;
  std::unordered_map<std::string, std::vector<seal::Modulus>> coeffmodulusmap_vars;
  for (auto n : vis.v) {
    coeffmodulusmap[n->getUniqueNodeId()] = coeff_modulus;
    if (dynamic_cast<Variable *>(n)) {
      coeffmodulusmap_vars[dynamic_cast<Variable &>(*n).getIdentifier()] = coeff_modulus;
    }
  }

  // modswitchinsertion visitor
  std::stringstream rr;
  InsertModSwitchVisitor
      modSwitchVis(rr, srv.getNoiseMap(), coeffmodulusmap, coeffmodulusmap_vars, calcInitNoiseHeuristic());

  // find modswitching nodes
  astProgram->accept(modSwitchVis);

  // binary expression where modSwitch is to be inserted
  auto binExprIns = modSwitchVis.getModSwitchNodes()[0];
  std::cout << modSwitchVis.getModSwitchNodes().size() << " insertions suggested" << std::endl;

  // do insert modswitch
  auto rewritten_ast = modSwitchVis.insertModSwitchInAst(&astProgram, binExprIns, coeffmodulusmap);


 // update coeff modulus map
  auto newCoeffmodulusmap = modSwitchVis.getCoeffModulusMap();
  auto newCoeffmodulusmap_vars = modSwitchVis.getCoeffModulusMapVars();

  std::cout << "AST after insertModswitch" << std::endl;

  // print rewritten AST
  std::stringstream sr;
  ProgramPrintVisitor q(sr);
  rewritten_ast->accept(q);
  std::cout << sr.str() << std::endl;

  auto fixMismatchVis = FixParamMismatchVisitor(newCoeffmodulusmap, newCoeffmodulusmap_vars);

  rewritten_ast->accept(fixMismatchVis);


  // print 'fixed' AST
  std::stringstream st;
  ProgramPrintVisitor m(st);
  rewritten_ast->accept(m);
  std::cout << st.str() << std::endl;

}


TEST_F(FixParameterMismatchVisitorVisitorTest, Bar_insert_modswitch_AST) {
/// bar.v: truncated

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
    )"""";
  auto astProgram = Parser::parse(std::string(program));

  // program's output
  const char *outputs = R""""(
      y = n276;
    )"""";

  auto astOutput = Parser::parse(std::string(outputs));
  // create and prepopulate TypeCheckingVisitor
  auto rootScope = std::make_unique<Scope>(*astProgram);

  registerInputVariable(*rootScope, "one", Datatype(Type::INT, true));
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

//
//  std::stringstream rs;
//  ProgramPrintVisitor p(rs);
//  astProgram->accept(p);
//  std::cout << rs.str() << std::endl;

  tcv->setRootScope(std::move(rootScope));
  astProgram->accept(*tcv);

  std::cout << " Running Program " << std::endl;

  // run the program and get its output
  auto map = tcv->getSecretTaintedNodes();
  RuntimeVisitor srv(*scf, *astInput, map);
  srv.executeAst(*astProgram);

  std::cout << "Getting all nodes." << std::endl;

  // Get nodes, but only expression nodes, not the block or return
  GetAllNodesVisitor vis;
  astProgram->accept(vis);

  //  map of coeff modulus vectors: we initially populate this map with the original coeff_modulus vector for each node in the AST
  auto coeff_modulus = scf->getContext().first_context_data()->parms().coeff_modulus();

  std::cout << "Populating coeff modulus map" << std::endl;

  // initially every node has the same ctxtmodulus vector
  std::unordered_map<std::string, std::vector<seal::Modulus>> coeffmodulusmap;
  std::unordered_map<std::string, std::vector<seal::Modulus>> coeffmodulusmap_vars;
  for (auto n : vis.v) {
    coeffmodulusmap[n->getUniqueNodeId()] = coeff_modulus;
    if (dynamic_cast<Variable *>(n)) {
      coeffmodulusmap_vars[dynamic_cast<Variable &>(*n).getIdentifier()] = coeff_modulus;
    }
  }

  std::stringstream ss;
  InsertModSwitchVisitor
      modSwitchVis(ss, srv.getNoiseMap(), coeffmodulusmap, coeffmodulusmap_vars, calcInitNoiseHeuristic());

  std::cout << "Looking for modswitch insertion sites" << std::endl;

  astProgram->accept(modSwitchVis); // find modswitching nodes

  if (!modSwitchVis.getModSwitchNodes().empty()) {
    std::cout << "Found " << modSwitchVis.getModSwitchNodes().size() << " potential modswitch sites" << std::endl;

    auto binExprIns = modSwitchVis.getModSwitchNodes(); //  modSwitches to be inserted

    auto rewritten_ast = modSwitchVis.insertModSwitchInAst(&astProgram, binExprIns[1], coeffmodulusmap);

    std::cout << "Inserted modswitches " << std::endl;

    auto newCoeffmodulusmap = modSwitchVis.getCoeffModulusMap();
    auto newCoeffmodulusmap_vars = modSwitchVis.getCoeffModulusMapVars();

    std::cout << "AST after insertModswitch" << std::endl;

    // print rewritten AST
    std::stringstream sr;
    ProgramPrintVisitor q(sr);
    rewritten_ast->accept(q);
    std::cout << sr.str() << std::endl;

    auto fixMismatchVis = FixParamMismatchVisitor(newCoeffmodulusmap, newCoeffmodulusmap_vars);

    rewritten_ast->accept(fixMismatchVis);

    std::cout << "AST after fixing param mismatches" << std::endl;

    // print 'fixed' AST
    std::stringstream st;
    ProgramPrintVisitor m(st);
    rewritten_ast->accept(m);
    std::cout << st.str() << std::endl;

  }
}



#endif