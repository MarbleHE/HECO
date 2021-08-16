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
#include "ast_opt/visitor/AvoidParamMismatchVisitor.h"

#include "gtest/gtest.h"
#ifdef HAVE_SEAL_BFV

class AvoidParameterMismatchVisitorVisitorTest : public ::testing::Test {

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

TEST_F(AvoidParameterMismatchVisitorVisitorTest, insertOne) {

  ///
  /// ((x^4 + y) * z^4) * w
  /// expected: insertion of a single modswitch operatioon applied to operands (x^4 + y) and z^4
  /// and an additional modswitch operation to w as result of the AvoidParameterMismatchVisitor

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
  for (auto n : vis.v) {
    coeffmodulusmap[n->getUniqueNodeId()] = coeff_modulus;
  }

  // modswitchinsertion visitor
  std::stringstream rr;
  InsertModSwitchVisitor modSwitchVis(rr, srv.getNoiseMap(), coeffmodulusmap, calcInitNoiseHeuristic());

  // find modswitching nodes
  astProgram->accept(modSwitchVis);

  // binary expression where modSwitch is to be inserted

  auto binExprIns = modSwitchVis.getModSwitchNodes()[0];
  std::cout << modSwitchVis.getModSwitchNodes().size() << " insertions suggested" << std::endl;

  // do insert modswitch
  auto rewritten_ast = modSwitchVis.insertModSwitchInAst(&astProgram, binExprIns, coeffmodulusmap);

  //update coeff modulus map
  modSwitchVis.updateCoeffModulusMap(binExprIns, 1);
  auto newCoeffmodulusmap = modSwitchVis.getCoeffModulusMap();

  // print updated coeff modulus map (sizes of entries = no. primes )
//  for (auto n : vis.v) {
//    std::cout << n->toString(false) << " " << n->getUniqueNodeId() << " " <<  newCoeffmodulusmap[n->getUniqueNodeId()].size() << std::endl;
//  }


  // There should be 7 primes at the binary expression whose children had a modswitch inserted
  EXPECT_EQ(newCoeffmodulusmap["BinaryExpression_120"].size(), 7);

//
  auto avoidMismatchVis = AvoidParamMismatchVisitor(newCoeffmodulusmap);
//
  astProgram->accept(avoidMismatchVis);
//
//  EXPECT_EQ(avoidMismatchVis.getModSwitchNodes().size(), 1);


}



#endif

