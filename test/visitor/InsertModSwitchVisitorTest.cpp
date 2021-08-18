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

class InsertModSwitchVisitorTest : public ::testing::Test {
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

TEST_F(InsertModSwitchVisitorTest, getModSwitchNodesTestOneFound) {

  ///
  /// (x^4 + y) * z^4
  /// expected: returns vector with last binary expression (corresponding to the last mult in the circuit)


  // program's input
  const char *inputs = R""""(
      secret int __input0__ = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
      secret int __input1__ = {24, 34, 222,   4,    1, 4,   9, 22, 1, 3};
      secret int __input2__ = {24, 34, 222,   4,    1, 4,   9, 22, 1, 3};
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
      secret int result = powx4plusy *** powz4;
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

  // initially every node has the same ctxtmodulus vector
  std::unordered_map<std::string, std::vector<seal::Modulus>> coeffmodulusmap;
  std::unordered_map<std::string, std::vector<seal::Modulus>> coeffmodulusmap_vars;
  for (auto n : vis.v) {
    coeffmodulusmap[n->getUniqueNodeId()] = coeff_modulus;
  }

  std::stringstream rr;
  InsertModSwitchVisitor modSwitchVis(rr, srv.getNoiseMap(), coeffmodulusmap, coeffmodulusmap_vars, calcInitNoiseHeuristic());

  astProgram->accept(modSwitchVis); // find modswitching nodes

  std::cout << modSwitchVis.getModSwitchNodes().size() << " potential modSwitch insertion site(s) found:" << std::endl;
  for (int j = 0; j < modSwitchVis.getModSwitchNodes().size(); j++) {
    std::cout << modSwitchVis.getModSwitchNodes()[j]->toString(false) << " "
              << modSwitchVis.getModSwitchNodes()[j]->getUniqueNodeId() << std::endl;
  }

  EXPECT_EQ(modSwitchVis.getModSwitchNodes()[0]->getUniqueNodeId(), "BinaryExpression_106");

}

TEST_F(InsertModSwitchVisitorTest, getModSwitchNodesTestNoneFound) {

  /// x^2 * x^2 : we don not expect a site where insertion of modswitch is possible.


// program's input
  const char *inputs = R""""(
      secret int __input0__ = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
      secret int __input1__ = {24, 34, 222,   4,    1, 4,   9, 22, 1, 3};
    )"""";
  auto astInput = Parser::parse(std::string(inputs));

  // program specification
  const char *program = R""""(
      secret int result = (__input0__ ***  __input0__) *** (__input1__ ***  __input1__);
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

  tcv->setRootScope(std::move(rootScope));
  astProgram->accept(*tcv);

  std::stringstream ss;
  ProgramPrintVisitor p(ss);
  astProgram->accept(p);
  std::cout << ss.str() << std::endl;

  // run the program and get its output
  auto map = tcv->getSecretTaintedNodes();
  RuntimeVisitor srv(*scf, *astInput, map);
  srv.executeAst(*astProgram);

  GetAllNodesVisitor vis;
  astProgram->accept(vis);


  //  map of coeff modulus vectors: we initially populate this map with the original coeff_modulus vector for each node in the AST
  auto coeff_modulus = scf->getContext().first_context_data()->parms().coeff_modulus();

  // initially every node has the same ctxtmodulus vector
  std::unordered_map<std::string, std::vector<seal::Modulus>> coeffmodulusmap;
  std::unordered_map<std::string, std::vector<seal::Modulus>> coeffmodulusmap_vars;
  for (auto n : vis.v) {
    coeffmodulusmap[n->getUniqueNodeId()] = coeff_modulus;
  }

  std::stringstream rr;
  InsertModSwitchVisitor modSwitchVis(rr, srv.getNoiseMap(), coeffmodulusmap, coeffmodulusmap_vars, calcInitNoiseHeuristic());

  astProgram->accept(modSwitchVis); // find modswitching nodes

  std::cout << modSwitchVis.getModSwitchNodes().size() << " potential modSwitch insertion site(s) found:" << std::endl;
  for (int j = 0; j < modSwitchVis.getModSwitchNodes().size(); j++) {
    std::cout << modSwitchVis.getModSwitchNodes()[j]->toString(false) << " "
              << modSwitchVis.getModSwitchNodes()[j]->getUniqueNodeId() << std::endl;
  }

  EXPECT_EQ(modSwitchVis.getModSwitchNodes().size(), 0);

}

TEST_F(InsertModSwitchVisitorTest, insertModSwitchTestNoChangeExpected) {
/// The circuit x^2 * x^2 remains unchanged

// program's input
  const char *inputs = R""""(
      secret int __input0__ = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
      secret int __input1__ = {24, 34, 222,   4,    1, 4,   9, 22, 1, 3};
    )"""";
  auto astInput = Parser::parse(std::string(inputs));

  // program specification
  const char *program = R""""(
      secret int result = (__input0__ ***  __input0__) *** (__input1__ ***  __input1__);
      return result;
    )"""";
  auto astProgram = Parser::parse(std::string(program));

  // program's output
  const char *outputs = R""""(
      y = result;
    )"""";
  auto astOutput = Parser::parse(std::string(outputs));


  // Keep a copy of o for later comparison
  auto astProgram_copy = astProgram->clone();

  // create and prepopulate TypeCheckingVisitor
  auto rootScope = std::make_unique<Scope>(*astProgram);
  registerInputVariable(*rootScope, "__input0__", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "__input1__", Datatype(Type::INT, true));

  tcv->setRootScope(std::move(rootScope));
  astProgram->accept(*tcv);

  // run the program and get its output
  auto map = tcv->getSecretTaintedNodes();
  RuntimeVisitor srv(*scf, *astInput, map);
  srv.executeAst(*astProgram);

  // Get nodes, but only expression nodes, not the block or return
  GetAllNodesVisitor vis;
  astProgram->accept(vis);

  //  map of coeff modulus vectors: we initially populate this map with the original coeff_modulus vector for each node in the AST
  auto coeff_modulus = scf->getContext().first_context_data()->parms().coeff_modulus();
  // initially every node has the same ctxtmodulus vector
  std::unordered_map<std::string, std::vector<seal::Modulus>> coeffmodulusmap;
  std::unordered_map<std::string, std::vector<seal::Modulus>> coeffmodulusmap_vars;
  for (auto n : vis.v) {
    coeffmodulusmap[n->getUniqueNodeId()] = coeff_modulus;
  }

  std::stringstream ss;
  InsertModSwitchVisitor modSwitchVis(ss, srv.getNoiseMap(), coeffmodulusmap, coeffmodulusmap_vars, calcInitNoiseHeuristic());

  astProgram->accept(modSwitchVis); // find modswitching nodes


  BinaryExpression *binExprIns = nullptr;

  if (!modSwitchVis.getModSwitchNodes().empty()) {
    binExprIns = modSwitchVis.getModSwitchNodes()[0]; //  modSwitches to be inserted
    std::cout << "TEST: " << binExprIns->getUniqueNodeId() << std::endl;
  }

  auto rewritten_ast = modSwitchVis.insertModSwitchInAst(&astProgram, binExprIns, coeffmodulusmap);

  std::stringstream rr;
  ProgramPrintVisitor p(rr);
  astProgram_copy->accept(p);
  std::cout << rr.str() << std::endl;


  //In this case, asts should be identical
  ASSERT_NE(rewritten_ast, nullptr);
  compareAST(*astProgram_copy, *rewritten_ast);
}

TEST_F(InsertModSwitchVisitorTest, insertModSwitchTestInsertionBeforeLastBinaryOpExpected) {

  ///
  /// (x^4 + y) * z^4
  /// expected: modSwitch ops inserted before last binary op
  // we also test if the noise_map is correctly recalculated and if the coeffmodulus map is correctly updated after modswitch insertion


  // program's input
  const char *inputs = R""""(
      secret int __input0__ = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
      secret int __input1__ = {24, 34, 222,   4,    1, 4,   9, 22, 1, 3};
      secret int __input2__ = {24, 34, 222,   4,    1, 4,   9, 22, 1, 3};
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
      secret int result = powx4plusy *** powz4;
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

  tcv->setRootScope(std::move(rootScope));
  astProgram->accept(*tcv);

  // run the program and get its output
  auto map = tcv->getSecretTaintedNodes();
  RuntimeVisitor srv(*scf, *astInput, map);
  srv.executeAst(*astProgram);

  // Get nodes, but only expression nodes, not the block or return
  GetAllNodesVisitor vis;
  astProgram->accept(vis);


  //  map of coeff modulus vectors: we initially populate this map with the original coeff_modulus vector for each node in the AST
  auto coeff_modulus = scf->getContext().first_context_data()->parms().coeff_modulus();

  // initially every node has the same ctxtmodulus vector
  std::unordered_map<std::string, std::vector<seal::Modulus>> coeffmodulusmap;
  std::unordered_map<std::string, std::vector<seal::Modulus>> coeffmodulusmap_vars;
  for (auto n : vis.v) {
    coeffmodulusmap[n->getUniqueNodeId()] = coeff_modulus;
  }

  std::stringstream ss;
  InsertModSwitchVisitor modSwitchVis(ss, srv.getNoiseMap(), coeffmodulusmap, coeffmodulusmap_vars, calcInitNoiseHeuristic());

  astProgram->accept(modSwitchVis); // find modswitching nodes

  auto binExprIns = modSwitchVis.getModSwitchNodes()[0]; //  modSwitches to be inserted

  auto rewritten_ast = modSwitchVis.insertModSwitchInAst(&astProgram, binExprIns, coeffmodulusmap);

  // update noise map
  modSwitchVis.updateNoiseMap(*rewritten_ast, &srv);

  //update coeff modulus map
  modSwitchVis.updateCoeffModulusMap(binExprIns, 1);
  coeffmodulusmap = modSwitchVis.getCoeffModulusMap();

  // print output program
  std::stringstream rr;
  ProgramPrintVisitor p(rr);
  rewritten_ast->accept(p);
  std::cout << rr.str() << std::endl;

  // expected program specification
  const char *expected_program = R""""(
      secret int powx2 = (__input0__ *** __input0__);
      secret int powx3 = (powx2 *** __input0__);
      secret int powx4 = (powx3 *** __input0__);
      secret int powx4plusy = (powx4 +++ __input1__);
      secret int powz2 = (__input2__ *** __input2__);
      secret int powz3 = (powz2 *** __input2__);
      secret int powz4 = (powz3 *** __input2__);
      secret int result = (modswitch(powx4plusy, 1) *** modswitch(powz4, 1));
      return result;
    )"""";
  auto astProgram_expected = Parser::parse(std::string(expected_program));


  //In this case, asts should be identical
  ASSERT_NE(rewritten_ast, nullptr);
  compareAST(*astProgram_expected, *rewritten_ast);

}

TEST_F(InsertModSwitchVisitorTest, insertModSwitchTestTwomodSwitchesBeforeLastBinaryOpExpected) {

  /// input:
  /// test circuit:    noise heuristics:     #primes in coeffmodulus
  /// x   x           32    32                 4     3
  ///  \ /             \    /                   \   /
  ///   x^2           result_noise                4
  ///
  /// We manipulate noise map and coeff modulus map in a way that two modSwitches are to be inserted after binaryOp.getLeft()
  ///
  /// Expected output program:
  /// {
  ///  secret int result = (modswitch(__input0__, 1) *** modswitch(__input1__, 2));
  ///  return result;
  /// }

  // program's input
  const char *inputs = R""""(
      secret int __input0__ = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
      secret int __input1__ = {24, 34, 222,   4,    1, 4,   9, 22, 1, 3};
    )"""";
  auto astInput = Parser::parse(std::string(inputs));

  // program specification
  const char *program = R""""(
      secret int result = (__input0__ ***  __input1__);
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

  tcv->setRootScope(std::move(rootScope));
  astProgram->accept(*tcv);

  std::stringstream ss;
  ProgramPrintVisitor p(ss);
  astProgram->accept(p);
  std::cout << ss.str() << std::endl;

  // run the program and get its output
  auto map = tcv->getSecretTaintedNodes();
  RuntimeVisitor srv(*scf, *astInput, map);
  srv.executeAst(*astProgram);

  GetAllNodesVisitor vis;
  astProgram->accept(vis);

  //  map of coeff modulus vectors: we initially populate this map with the original coeff_modulus vector for each node in the AST
  auto coeff_modulus = scf->getContext().first_context_data()->parms().coeff_modulus();

  // initially every node has the same ctxtmodulus vector
  std::unordered_map<std::string, std::vector<seal::Modulus>> coeffmodulusmap;
  std::unordered_map<std::string, std::vector<seal::Modulus>> coeffmodulusmap_vars;
  for (auto n : vis.v) {
    coeffmodulusmap[n->getUniqueNodeId()] = coeff_modulus;
  }

  // remove the last prime for binaryOp.getLeft() in coeffmodulus map (our goal is to have two modswitches inserted...)
  coeffmodulusmap["Variable_33"].pop_back();

  std::cout << "Initial Noise Heur: " << calcInitNoiseHeuristic() << std::endl;

  auto tamperedNoiseMap = srv.getNoiseMap();
  tamperedNoiseMap["Variable_33"] = 32;
  tamperedNoiseMap["Variable_35"] = 32;

  std::stringstream rr;
  InsertModSwitchVisitor modSwitchVis(rr, tamperedNoiseMap, coeffmodulusmap, coeffmodulusmap_vars, calcInitNoiseHeuristic());
  astProgram->accept(modSwitchVis); // find modswitching nodes

  std::cout << modSwitchVis.getModSwitchNodes().size() << " potential modSwitch insertion site(s) found:" << std::endl;

  auto binExprIns = modSwitchVis.getModSwitchNodes()[0]; //  modSwitches to be inserted

  auto rewritten_ast = modSwitchVis.insertModSwitchInAst(&astProgram, binExprIns, coeffmodulusmap);

  std::cout << "Resulting AST:" << std::endl;

  std::stringstream rs;
  ProgramPrintVisitor q(rs);
  rewritten_ast->accept(q);
  std::cout << rs.str() << std::endl;

  const char *expected_program = R""""(
     secret int result = (modswitch(__input0__, 1) *** modswitch(__input1__, 2));
     return result;
    )"""";
  auto expected_astProgram = Parser::parse(std::string(expected_program));

  compareAST(*rewritten_ast, *expected_astProgram);
}

TEST_F(InsertModSwitchVisitorTest, removeModSwitchTest) {
  /// we expect removing of the modswitches

  // program's input
  const char *inputs = R""""(
      secret int __input0__ = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
      secret int __input1__ = {24, 34, 222,   4,    1, 4,   9, 22, 1, 3};
    )"""";
  auto astInput = Parser::parse(std::string(inputs));

  // program specification
  const char *program = R""""(
      secret int result = (modswitch(__input0__, 1) *** modswitch(__input1__, 2));
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

  // run the program and get its output
  auto map = tcv->getSecretTaintedNodes();
  RuntimeVisitor srv(*scf, *astInput, map);
  srv.executeAst(*astProgram);

  // Get nodes, but only expression nodes, not the block or return
  GetAllNodesVisitor vis;
  astProgram->accept(vis);


  //  map of coeff modulus vectors: we initially populate this map with the original coeff_modulus vector for each node in the AST
  auto coeff_modulus = scf->getContext().first_context_data()->parms().coeff_modulus();

  // initially every node has the same ctxtmodulus vector
  std::unordered_map<std::string, std::vector<seal::Modulus>> coeffmodulusmap;
  std::unordered_map<std::string, std::vector<seal::Modulus>> coeffmodulusmap_vars;
  for (auto n : vis.v) {
    coeffmodulusmap[n->getUniqueNodeId()] = coeff_modulus;
  }

  std::stringstream ss;
  InsertModSwitchVisitor modSwitchVis(ss, srv.getNoiseMap(), coeffmodulusmap, coeffmodulusmap_vars, calcInitNoiseHeuristic());

//  int index = 0;
//  for (auto n : vis.v) {
//    std::cout << "index: " << index << n->toString(false) << std::endl;
//    index++;
//  }

  auto astRemoved = modSwitchVis.removeModSwitchFromAst(&astProgram, dynamic_cast<BinaryExpression *>(vis.v[3]));

  std::stringstream rr;
  ProgramPrintVisitor p(rr);
  astRemoved->accept(p);
  std::cout << rr.str() << std::endl;

  // WANT: remove BinaryExpression_40
  // TODO: remove modswitches


  //expected circuit
  // program specification
  const char *expected_program = R""""(
      secret int result = (__input0__ ***  __input1__);
      return result;
    )"""";
  auto expected_astProgram = Parser::parse(std::string(expected_program));

  compareAST(*astRemoved, *expected_astProgram);

}

TEST_F(InsertModSwitchVisitorTest, rewriteAstTest) {

  ///
  /// (x^4 + y) * z^4
  /// expected: modSwitch ops inserted before last binary op
  // we also test if the noise_map is correctly recalculated and if the coeffmodulus map is correctly updated after modswitch insertion


  // program's input
  const char *inputs = R""""(
      secret int __input0__ = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
      secret int __input1__ = {24, 34, 222,   4,    1, 4,   9, 22, 1, 3};
      secret int __input2__ = {24, 34, 222,   4,    1, 4,   9, 22, 1, 3};
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
      secret int result = powx4plusy *** powz4;
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

  tcv->setRootScope(std::move(rootScope));
  astProgram->accept(*tcv);

  // run the program and get its output
  auto map = tcv->getSecretTaintedNodes();
  RuntimeVisitor srv(*scf, *astInput, map);
  srv.executeAst(*astProgram);

  // Get nodes, but only expression nodes, not the block or return
  GetAllNodesVisitor vis;
  astProgram->accept(vis);


  //  map of coeff modulus vectors: we initially populate this map with the original coeff_modulus vector for each node in the AST
  auto coeff_modulus = scf->getContext().first_context_data()->parms().coeff_modulus();

  // initially every node has the same ctxtmodulus vector
  std::unordered_map<std::string, std::vector<seal::Modulus>> coeffmodulusmap;
  std::unordered_map<std::string, std::vector<seal::Modulus>> coeffmodulusmap_vars;
  for (auto n : vis.v) {
    coeffmodulusmap[n->getUniqueNodeId()] = coeff_modulus;
  }

  std::stringstream ss;
  InsertModSwitchVisitor modSwitchVis(ss, srv.getNoiseMap(), coeffmodulusmap, coeffmodulusmap_vars, calcInitNoiseHeuristic());

 // auto rewrittenAst = modSwitchVis.rewriteAst(&astProgram, srv, coeffmodulusmap); // TODO: figure out how the srv stuff works

}


/// -------- EPFL circuits --------

/// add.v

TEST_F(InsertModSwitchVisitorTest, Adder_insert_modswitch_AST) {
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
secret int b0 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b1 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b2 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b3 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b4 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b5 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b6 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b7 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b8 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b9 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b10 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b11 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b12 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b13 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b14 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b15 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b16 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b17 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b18 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b19 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b20 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b21 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b22 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b23 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b24 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b25 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b26 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b27 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b28 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b29 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b30 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b31 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b32 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b33 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b34 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b35 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b36 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b37 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b38 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b39 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b40 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b41 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b42 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b43 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b44 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b45 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b46 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b47 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b48 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b49 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b50 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b51 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b52 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b53 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b54 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b55 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b56 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b57 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b58 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b59 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b60 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b61 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b62 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b63 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b64 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b65 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b66 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b67 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b68 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b69 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b70 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b71 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b72 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b73 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b74 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b75 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b76 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b77 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b78 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b79 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b80 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b81 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b82 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b83 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b84 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b85 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b86 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b87 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b88 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b89 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b90 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b91 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b92 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b93 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b94 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b95 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b96 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b97 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b98 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b99 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b100 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b101 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b102 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b103 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b104 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b105 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b106 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b107 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b108 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b109 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b110 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b111 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b112 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b113 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b114 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b115 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b116 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b117 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b118 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b119 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b120 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b121 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b122 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b123 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b124 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b125 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b126 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b127 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f0 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f1 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f2 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f3 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f4 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f5 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f6 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f7 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f8 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f9 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f10 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f11 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f12 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f13 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f14 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f15 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f16 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f17 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f18 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f19 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f20 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f21 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f22 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f23 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f24 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f25 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f26 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f27 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f28 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f29 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f30 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f31 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f32 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f33 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f34 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f35 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f36 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f37 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f38 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f39 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f40 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f41 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f42 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f43 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f44 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f45 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f46 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f47 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f48 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f49 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f50 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f51 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f52 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f53 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f54 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f55 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f56 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f57 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f58 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f59 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f60 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f61 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f62 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f63 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f64 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f65 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f66 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f67 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f68 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f69 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f70 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f71 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f72 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f73 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f74 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f75 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f76 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f77 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f78 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f79 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f80 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f81 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f82 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f83 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f84 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f85 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f86 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f87 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f88 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f89 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f90 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f91 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f92 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f93 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f94 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f95 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f96 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f97 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f98 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f99 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f100 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f101 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f102 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f103 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f104 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f105 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f106 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f107 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f108 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f109 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f110 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f111 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f112 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f113 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f114 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f115 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f116 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f117 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f118 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f119 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f120 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f121 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f122 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f123 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f124 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f125 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f126 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f127 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int cOut = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int one = {1,  1,   1,   1,  1, 1, 1,  1, 1, 1};
    )"""";
  auto astInput = Parser::parse(std::string(inputs));

  // program specification
  const char *program = R""""(
secret int n386 = a0 *** (one --- b0);
secret int n387 = (one --- a0) *** b0;
secret int f0 = (n386 +++ n387 --- n386 *** n387);
secret int n389 = a0 *** b0;
secret int n390 = (one --- a1) *** (one --- b1);
secret int n391 = a1 *** b1;
secret int n392 = (one --- n390) *** (one --- n391);
secret int n393 = n389 *** (one --- n392);
secret int n394 = (one --- n389) *** n392;
secret int f1 = (n393 +++ n394 --- n393 *** n394);
secret int n396 = n389 *** (one --- n390);
secret int n397 = (one --- n391) *** (one --- n396);
secret int n398 = (one --- a2) *** (one --- b2);
secret int n399 = a2 *** b2;
secret int n400 = (one --- n398) *** (one --- n399);
secret int n401 = n397 *** (one --- n400);
secret int n402 = (one --- n397) *** n400;
secret int f2 = (one --- n401) *** (one --- n402);
secret int n404 = (one --- n397) *** (one --- n398);
secret int n405 = (one --- n399) *** (one --- n404);
secret int n406 = (one --- a3) *** (one --- b3);
secret int n407 = a3 *** b3;
secret int n408 = (one --- n406) *** (one --- n407);
secret int n409 = n405 *** (one --- n408);
secret int n410 = (one --- n405) *** n408;
secret int f3 = (one --- n409) *** (one --- n410);
secret int n412 = (one --- n405) *** (one --- n406);
secret int n413 = (one --- n407) *** (one --- n412);
secret int n414 = (one --- a4) *** (one --- b4);
secret int n415 = a4 *** b4;
secret int n416 = (one --- n414) *** (one --- n415);
secret int n417 = n413 *** (one --- n416);
secret int n418 = (one --- n413) *** n416;
secret int f4 = (one --- n417) *** (one --- n418);
secret int n420 = (one --- n413) *** (one --- n414);
secret int n421 = (one --- n415) *** (one --- n420);
secret int n422 = (one --- a5) *** (one --- b5);
secret int n423 = a5 *** b5;
secret int n424 = (one --- n422) *** (one --- n423);
secret int n425 = n421 *** (one --- n424);
secret int n426 = (one --- n421) *** n424;
secret int f5 = (one --- n425) *** (one --- n426);
secret int n428 = (one --- n421) *** (one --- n422);
secret int n429 = (one --- n423) *** (one --- n428);
secret int n430 = (one --- a6) *** (one --- b6);
secret int n431 = a6 *** b6;
secret int n432 = (one --- n430) *** (one --- n431);
secret int n433 = n429 *** (one --- n432);
secret int n434 = (one --- n429) *** n432;
secret int f6 = (one --- n433) *** (one --- n434);
secret int n436 = (one --- n429) *** (one --- n430);
secret int n437 = (one --- n431) *** (one --- n436);
secret int n438 = (one --- a7) *** (one --- b7);
secret int n439 = a7 *** b7;
secret int n440 = (one --- n438) *** (one --- n439);
secret int n441 = n437 *** (one --- n440);
secret int n442 = (one --- n437) *** n440;
secret int f7 = (one --- n441) *** (one --- n442);
secret int n444 = (one --- n437) *** (one --- n438);
secret int n445 = (one --- n439) *** (one --- n444);
secret int n446 = (one --- a8) *** (one --- b8);
secret int n447 = a8 *** b8;
secret int n448 = (one --- n446) *** (one --- n447);
secret int n449 = n445 *** (one --- n448);
secret int n450 = (one --- n445) *** n448;
secret int f8 = (one --- n449) *** (one --- n450);
secret int n452 = (one --- n445) *** (one --- n446);
secret int n453 = (one --- n447) *** (one --- n452);
secret int n454 = (one --- a9) *** (one --- b9);
secret int n455 = a9 *** b9;
secret int n456 = (one --- n454) *** (one --- n455);
secret int n457 = n453 *** (one --- n456);
secret int n458 = (one --- n453) *** n456;
secret int f9 = (one --- n457) *** (one --- n458);
secret int n460 = (one --- n453) *** (one --- n454);
secret int n461 = (one --- n455) *** (one --- n460);
secret int n462 = (one --- a10) *** (one --- b10);
secret int n463 = a10 *** b10;
secret int n464 = (one --- n462) *** (one --- n463);
secret int n465 = n461 *** (one --- n464);
secret int n466 = (one --- n461) *** n464;
secret int f10 = (one --- n465) *** (one --- n466);
secret int n468 = (one --- n461) *** (one --- n462);
secret int n469 = (one --- n463) *** (one --- n468);
secret int n470 = (one --- a11) *** (one --- b11);
secret int n471 = a11 *** b11;
secret int n472 = (one --- n470) *** (one --- n471);
secret int n473 = n469 *** (one --- n472);
secret int n474 = (one --- n469) *** n472;
secret int f11 = (one --- n473) *** (one --- n474);
secret int n476 = (one --- n469) *** (one --- n470);
secret int n477 = (one --- n471) *** (one --- n476);
secret int n478 = (one --- a12) *** (one --- b12);
secret int n479 = a12 *** b12;
secret int n480 = (one --- n478) *** (one --- n479);
secret int n481 = n477 *** (one --- n480);
secret int n482 = (one --- n477) *** n480;
secret int f12 = (one --- n481) *** (one --- n482);
secret int n484 = (one --- n477) *** (one --- n478);
secret int n485 = (one --- n479) *** (one --- n484);
secret int n486 = (one --- a13) *** (one --- b13);
secret int n487 = a13 *** b13;
secret int n488 = (one --- n486) *** (one --- n487);
secret int n489 = n485 *** (one --- n488);
secret int n490 = (one --- n485) *** n488;
secret int f13 = (one --- n489) *** (one --- n490);
secret int n492 = (one --- n485) *** (one --- n486);
secret int n493 = (one --- n487) *** (one --- n492);
secret int n494 = (one --- a14) *** (one --- b14);
secret int n495 = a14 *** b14;
secret int n496 = (one --- n494) *** (one --- n495);
secret int n497 = n493 *** (one --- n496);
secret int n498 = (one --- n493) *** n496;
secret int f14 = (one --- n497) *** (one --- n498);
secret int n500 = (one --- n493) *** (one --- n494);
secret int n501 = (one --- n495) *** (one --- n500);
secret int n502 = (one --- a15) *** (one --- b15);
secret int n503 = a15 *** b15;
secret int n504 = (one --- n502) *** (one --- n503);
secret int n505 = n501 *** (one --- n504);
secret int n506 = (one --- n501) *** n504;
secret int f15 = (one --- n505) *** (one --- n506);
secret int n508 = (one --- n501) *** (one --- n502);
secret int n509 = (one --- n503) *** (one --- n508);
secret int n510 = (one --- a16) *** (one --- b16);
secret int n511 = a16 *** b16;
secret int n512 = (one --- n510) *** (one --- n511);
secret int n513 = n509 *** (one --- n512);
secret int n514 = (one --- n509) *** n512;
secret int f16 = (one --- n513) *** (one --- n514);
secret int n516 = (one --- n509) *** (one --- n510);
secret int n517 = (one --- n511) *** (one --- n516);
secret int n518 = (one --- a17) *** (one --- b17);
secret int n519 = a17 *** b17;
secret int n520 = (one --- n518) *** (one --- n519);
secret int n521 = n517 *** (one --- n520);
secret int n522 = (one --- n517) *** n520;
secret int f17 = (one --- n521) *** (one --- n522);
secret int n524 = (one --- n517) *** (one --- n518);
secret int n525 = (one --- n519) *** (one --- n524);
secret int n526 = (one --- a18) *** (one --- b18);
secret int n527 = a18 *** b18;
secret int n528 = (one --- n526) *** (one --- n527);
secret int n529 = n525 *** (one --- n528);
secret int n530 = (one --- n525) *** n528;
secret int f18 = (one --- n529) *** (one --- n530);
secret int n532 = (one --- n525) *** (one --- n526);
secret int n533 = (one --- n527) *** (one --- n532);
secret int n534 = (one --- a19) *** (one --- b19);
secret int n535 = a19 *** b19;
secret int n536 = (one --- n534) *** (one --- n535);
secret int n537 = n533 *** (one --- n536);
secret int n538 = (one --- n533) *** n536;
secret int f19 = (one --- n537) *** (one --- n538);
secret int n540 = (one --- n533) *** (one --- n534);
secret int n541 = (one --- n535) *** (one --- n540);
secret int n542 = (one --- a20) *** (one --- b20);
secret int n543 = a20 *** b20;
secret int n544 = (one --- n542) *** (one --- n543);
secret int n545 = n541 *** (one --- n544);
secret int n546 = (one --- n541) *** n544;
secret int f20 = (one --- n545) *** (one --- n546);
secret int n548 = (one --- n541) *** (one --- n542);
secret int n549 = (one --- n543) *** (one --- n548);
secret int n550 = (one --- a21) *** (one --- b21);
secret int n551 = a21 *** b21;
secret int n552 = (one --- n550) *** (one --- n551);
secret int n553 = n549 *** (one --- n552);
secret int n554 = (one --- n549) *** n552;
secret int f21 = (one --- n553) *** (one --- n554);
secret int n556 = (one --- n549) *** (one --- n550);
secret int n557 = (one --- n551) *** (one --- n556);
secret int n558 = (one --- a22) *** (one --- b22);
secret int n559 = a22 *** b22;
secret int n560 = (one --- n558) *** (one --- n559);
secret int n561 = n557 *** (one --- n560);
secret int n562 = (one --- n557) *** n560;
secret int f22 = (one --- n561) *** (one --- n562);
secret int n564 = (one --- n557) *** (one --- n558);
secret int n565 = (one --- n559) *** (one --- n564);
secret int n566 = (one --- a23) *** (one --- b23);
secret int n567 = a23 *** b23;
secret int n568 = (one --- n566) *** (one --- n567);
secret int n569 = n565 *** (one --- n568);
secret int n570 = (one --- n565) *** n568;
secret int f23 = (one --- n569) *** (one --- n570);
secret int n572 = (one --- n565) *** (one --- n566);
secret int n573 = (one --- n567) *** (one --- n572);
secret int n574 = (one --- a24) *** (one --- b24);
secret int n575 = a24 *** b24;
secret int n576 = (one --- n574) *** (one --- n575);
secret int n577 = n573 *** (one --- n576);
secret int n578 = (one --- n573) *** n576;
secret int f24 = (one --- n577) *** (one --- n578);
secret int n580 = (one --- n573) *** (one --- n574);
secret int n581 = (one --- n575) *** (one --- n580);
secret int n582 = (one --- a25) *** (one --- b25);
secret int n583 = a25 *** b25;
secret int n584 = (one --- n582) *** (one --- n583);
secret int n585 = n581 *** (one --- n584);
secret int n586 = (one --- n581) *** n584;
secret int f25 = (one --- n585) *** (one --- n586);
secret int n588 = (one --- n581) *** (one --- n582);
secret int n589 = (one --- n583) *** (one --- n588);
secret int n590 = (one --- a26) *** (one --- b26);
secret int n591 = a26 *** b26;
secret int n592 = (one --- n590) *** (one --- n591);
secret int n593 = n589 *** (one --- n592);
secret int n594 = (one --- n589) *** n592;
secret int f26 = (one --- n593) *** (one --- n594);
secret int n596 = (one --- n589) *** (one --- n590);
secret int n597 = (one --- n591) *** (one --- n596);
secret int n598 = (one --- a27) *** (one --- b27);
secret int n599 = a27 *** b27;
secret int n600 = (one --- n598) *** (one --- n599);
secret int n601 = n597 *** (one --- n600);
secret int n602 = (one --- n597) *** n600;
secret int f27 = (one --- n601) *** (one --- n602);
secret int n604 = (one --- n597) *** (one --- n598);
secret int n605 = (one --- n599) *** (one --- n604);
secret int n606 = (one --- a28) *** (one --- b28);
secret int n607 = a28 *** b28;
secret int n608 = (one --- n606) *** (one --- n607);
secret int n609 = n605 *** (one --- n608);
secret int n610 = (one --- n605) *** n608;
secret int f28 = (one --- n609) *** (one --- n610);
secret int n612 = (one --- n605) *** (one --- n606);
secret int n613 = (one --- n607) *** (one --- n612);
secret int n614 = (one --- a29) *** (one --- b29);
secret int n615 = a29 *** b29;
secret int n616 = (one --- n614) *** (one --- n615);
secret int n617 = n613 *** (one --- n616);
secret int n618 = (one --- n613) *** n616;
secret int f29 = (one --- n617) *** (one --- n618);
secret int n620 = (one --- n613) *** (one --- n614);
secret int n621 = (one --- n615) *** (one --- n620);
secret int n622 = (one --- a30) *** (one --- b30);
secret int n623 = a30 *** b30;
secret int n624 = (one --- n622) *** (one --- n623);
secret int n625 = n621 *** (one --- n624);
secret int n626 = (one --- n621) *** n624;
secret int f30 = (one --- n625) *** (one --- n626);
secret int n628 = (one --- n621) *** (one --- n622);
secret int n629 = (one --- n623) *** (one --- n628);
secret int n630 = (one --- a31) *** (one --- b31);
secret int n631 = a31 *** b31;
secret int n632 = (one --- n630) *** (one --- n631);
secret int n633 = n629 *** (one --- n632);
secret int n634 = (one --- n629) *** n632;
secret int f31 = (one --- n633) *** (one --- n634);
secret int n636 = (one --- n629) *** (one --- n630);
secret int n637 = (one --- n631) *** (one --- n636);
secret int n638 = (one --- a32) *** (one --- b32);
secret int n639 = a32 *** b32;
secret int n640 = (one --- n638) *** (one --- n639);
secret int n641 = n637 *** (one --- n640);
secret int n642 = (one --- n637) *** n640;
secret int f32 = (one --- n641) *** (one --- n642);
secret int n644 = (one --- n637) *** (one --- n638);
secret int n645 = (one --- n639) *** (one --- n644);
secret int n646 = (one --- a33) *** (one --- b33);
secret int n647 = a33 *** b33;
secret int n648 = (one --- n646) *** (one --- n647);
secret int n649 = n645 *** (one --- n648);
secret int n650 = (one --- n645) *** n648;
secret int f33 = (one --- n649) *** (one --- n650);
secret int n652 = (one --- n645) *** (one --- n646);
secret int n653 = (one --- n647) *** (one --- n652);
secret int n654 = (one --- a34) *** (one --- b34);
secret int n655 = a34 *** b34;
secret int n656 = (one --- n654) *** (one --- n655);
secret int n657 = n653 *** (one --- n656);
secret int n658 = (one --- n653) *** n656;
secret int f34 = (one --- n657) *** (one --- n658);
secret int n660 = (one --- n653) *** (one --- n654);
secret int n661 = (one --- n655) *** (one --- n660);
secret int n662 = (one --- a35) *** (one --- b35);
secret int n663 = a35 *** b35;
secret int n664 = (one --- n662) *** (one --- n663);
secret int n665 = n661 *** (one --- n664);
secret int n666 = (one --- n661) *** n664;
secret int f35 = (one --- n665) *** (one --- n666);
secret int n668 = (one --- n661) *** (one --- n662);
secret int n669 = (one --- n663) *** (one --- n668);
secret int n670 = (one --- a36) *** (one --- b36);
secret int n671 = a36 *** b36;
secret int n672 = (one --- n670) *** (one --- n671);
secret int n673 = n669 *** (one --- n672);
secret int n674 = (one --- n669) *** n672;
secret int f36 = (one --- n673) *** (one --- n674);
secret int n676 = (one --- n669) *** (one --- n670);
secret int n677 = (one --- n671) *** (one --- n676);
secret int n678 = (one --- a37) *** (one --- b37);
secret int n679 = a37 *** b37;
secret int n680 = (one --- n678) *** (one --- n679);
secret int n681 = n677 *** (one --- n680);
secret int n682 = (one --- n677) *** n680;
secret int f37 = (one --- n681) *** (one --- n682);
secret int n684 = (one --- n677) *** (one --- n678);
secret int n685 = (one --- n679) *** (one --- n684);
secret int n686 = (one --- a38) *** (one --- b38);
secret int n687 = a38 *** b38;
secret int n688 = (one --- n686) *** (one --- n687);
secret int n689 = n685 *** (one --- n688);
secret int n690 = (one --- n685) *** n688;
secret int f38 = (one --- n689) *** (one --- n690);
secret int n692 = (one --- n685) *** (one --- n686);
secret int n693 = (one --- n687) *** (one --- n692);
secret int n694 = (one --- a39) *** (one --- b39);
secret int n695 = a39 *** b39;
secret int n696 = (one --- n694) *** (one --- n695);
secret int n697 = n693 *** (one --- n696);
secret int n698 = (one --- n693) *** n696;
secret int f39 = (one --- n697) *** (one --- n698);
secret int n700 = (one --- n693) *** (one --- n694);
secret int n701 = (one --- n695) *** (one --- n700);
secret int n702 = (one --- a40) *** (one --- b40);
secret int n703 = a40 *** b40;
secret int n704 = (one --- n702) *** (one --- n703);
secret int n705 = n701 *** (one --- n704);
secret int n706 = (one --- n701) *** n704;
secret int f40 = (one --- n705) *** (one --- n706);
secret int n708 = (one --- n701) *** (one --- n702);
secret int n709 = (one --- n703) *** (one --- n708);
secret int n710 = (one --- a41) *** (one --- b41);
secret int n711 = a41 *** b41;
secret int n712 = (one --- n710) *** (one --- n711);
secret int n713 = n709 *** (one --- n712);
secret int n714 = (one --- n709) *** n712;
secret int f41 = (one --- n713) *** (one --- n714);
secret int n716 = (one --- n709) *** (one --- n710);
secret int n717 = (one --- n711) *** (one --- n716);
secret int n718 = (one --- a42) *** (one --- b42);
secret int n719 = a42 *** b42;
secret int n720 = (one --- n718) *** (one --- n719);
secret int n721 = n717 *** (one --- n720);
secret int n722 = (one --- n717) *** n720;
secret int f42 = (one --- n721) *** (one --- n722);
secret int n724 = (one --- n717) *** (one --- n718);
secret int n725 = (one --- n719) *** (one --- n724);
secret int n726 = (one --- a43) *** (one --- b43);
secret int n727 = a43 *** b43;
secret int n728 = (one --- n726) *** (one --- n727);
secret int n729 = n725 *** (one --- n728);
secret int n730 = (one --- n725) *** n728;
secret int f43 = (one --- n729) *** (one --- n730);
secret int n732 = (one --- n725) *** (one --- n726);
secret int n733 = (one --- n727) *** (one --- n732);
secret int n734 = (one --- a44) *** (one --- b44);
secret int n735 = a44 *** b44;
secret int n736 = (one --- n734) *** (one --- n735);
secret int n737 = n733 *** (one --- n736);
secret int n738 = (one --- n733) *** n736;
secret int f44 = (one --- n737) *** (one --- n738);
secret int n740 = (one --- n733) *** (one --- n734);
secret int n741 = (one --- n735) *** (one --- n740);
secret int n742 = (one --- a45) *** (one --- b45);
secret int n743 = a45 *** b45;
secret int n744 = (one --- n742) *** (one --- n743);
secret int n745 = n741 *** (one --- n744);
secret int n746 = (one --- n741) *** n744;
secret int f45 = (one --- n745) *** (one --- n746);
secret int n748 = (one --- n741) *** (one --- n742);
secret int n749 = (one --- n743) *** (one --- n748);
secret int n750 = (one --- a46) *** (one --- b46);
secret int n751 = a46 *** b46;
secret int n752 = (one --- n750) *** (one --- n751);
secret int n753 = n749 *** (one --- n752);
secret int n754 = (one --- n749) *** n752;
secret int f46 = (one --- n753) *** (one --- n754);
secret int n756 = (one --- n749) *** (one --- n750);
secret int n757 = (one --- n751) *** (one --- n756);
secret int n758 = (one --- a47) *** (one --- b47);
secret int n759 = a47 *** b47;
secret int n760 = (one --- n758) *** (one --- n759);
secret int n761 = n757 *** (one --- n760);
secret int n762 = (one --- n757) *** n760;
secret int f47 = (one --- n761) *** (one --- n762);
secret int n764 = (one --- n757) *** (one --- n758);
secret int n765 = (one --- n759) *** (one --- n764);
secret int n766 = (one --- a48) *** (one --- b48);
secret int n767 = a48 *** b48;
secret int n768 = (one --- n766) *** (one --- n767);
secret int n769 = n765 *** (one --- n768);
secret int n770 = (one --- n765) *** n768;
secret int f48 = (one --- n769) *** (one --- n770);
secret int n772 = (one --- n765) *** (one --- n766);
secret int n773 = (one --- n767) *** (one --- n772);
secret int n774 = (one --- a49) *** (one --- b49);
secret int n775 = a49 *** b49;
secret int n776 = (one --- n774) *** (one --- n775);
secret int n777 = n773 *** (one --- n776);
secret int n778 = (one --- n773) *** n776;
secret int f49 = (one --- n777) *** (one --- n778);
secret int n780 = (one --- n773) *** (one --- n774);
secret int n781 = (one --- n775) *** (one --- n780);
secret int n782 = (one --- a50) *** (one --- b50);
secret int n783 = a50 *** b50;
secret int n784 = (one --- n782) *** (one --- n783);
secret int n785 = n781 *** (one --- n784);
secret int n786 = (one --- n781) *** n784;
secret int f50 = (one --- n785) *** (one --- n786);
secret int n788 = (one --- n781) *** (one --- n782);
secret int n789 = (one --- n783) *** (one --- n788);
secret int n790 = (one --- a51) *** (one --- b51);
secret int n791 = a51 *** b51;
secret int n792 = (one --- n790) *** (one --- n791);
secret int n793 = n789 *** (one --- n792);
secret int n794 = (one --- n789) *** n792;
secret int f51 = (one --- n793) *** (one --- n794);
secret int n796 = (one --- n789) *** (one --- n790);
secret int n797 = (one --- n791) *** (one --- n796);
secret int n798 = (one --- a52) *** (one --- b52);
secret int n799 = a52 *** b52;
secret int n800 = (one --- n798) *** (one --- n799);
secret int n801 = n797 *** (one --- n800);
secret int n802 = (one --- n797) *** n800;
secret int f52 = (one --- n801) *** (one --- n802);
secret int n804 = (one --- n797) *** (one --- n798);
secret int n805 = (one --- n799) *** (one --- n804);
secret int n806 = (one --- a53) *** (one --- b53);
secret int n807 = a53 *** b53;
secret int n808 = (one --- n806) *** (one --- n807);
secret int n809 = n805 *** (one --- n808);
secret int n810 = (one --- n805) *** n808;
secret int f53 = (one --- n809) *** (one --- n810);
secret int n812 = (one --- n805) *** (one --- n806);
secret int n813 = (one --- n807) *** (one --- n812);
secret int n814 = (one --- a54) *** (one --- b54);
secret int n815 = a54 *** b54;
secret int n816 = (one --- n814) *** (one --- n815);
secret int n817 = n813 *** (one --- n816);
secret int n818 = (one --- n813) *** n816;
secret int f54 = (one --- n817) *** (one --- n818);
secret int n820 = (one --- n813) *** (one --- n814);
secret int n821 = (one --- n815) *** (one --- n820);
secret int n822 = (one --- a55) *** (one --- b55);
secret int n823 = a55 *** b55;
secret int n824 = (one --- n822) *** (one --- n823);
secret int n825 = n821 *** (one --- n824);
secret int n826 = (one --- n821) *** n824;
secret int f55 = (one --- n825) *** (one --- n826);
secret int n828 = (one --- n821) *** (one --- n822);
secret int n829 = (one --- n823) *** (one --- n828);
secret int n830 = (one --- a56) *** (one --- b56);
secret int n831 = a56 *** b56;
secret int n832 = (one --- n830) *** (one --- n831);
secret int n833 = n829 *** (one --- n832);
secret int n834 = (one --- n829) *** n832;
secret int f56 = (one --- n833) *** (one --- n834);
secret int n836 = (one --- n829) *** (one --- n830);
secret int n837 = (one --- n831) *** (one --- n836);
secret int n838 = (one --- a57) *** (one --- b57);
secret int n839 = a57 *** b57;
secret int n840 = (one --- n838) *** (one --- n839);
secret int n841 = n837 *** (one --- n840);
secret int n842 = (one --- n837) *** n840;
secret int f57 = (one --- n841) *** (one --- n842);
secret int n844 = (one --- n837) *** (one --- n838);
secret int n845 = (one --- n839) *** (one --- n844);
secret int n846 = (one --- a58) *** (one --- b58);
secret int n847 = a58 *** b58;
secret int n848 = (one --- n846) *** (one --- n847);
secret int n849 = n845 *** (one --- n848);
secret int n850 = (one --- n845) *** n848;
secret int f58 = (one --- n849) *** (one --- n850);
secret int n852 = (one --- n845) *** (one --- n846);
secret int n853 = (one --- n847) *** (one --- n852);
secret int n854 = (one --- a59) *** (one --- b59);
secret int n855 = a59 *** b59;
secret int n856 = (one --- n854) *** (one --- n855);
secret int n857 = n853 *** (one --- n856);
secret int n858 = (one --- n853) *** n856;
secret int f59 = (one --- n857) *** (one --- n858);
secret int n860 = (one --- n853) *** (one --- n854);
secret int n861 = (one --- n855) *** (one --- n860);
secret int n862 = (one --- a60) *** (one --- b60);
secret int n863 = a60 *** b60;
secret int n864 = (one --- n862) *** (one --- n863);
secret int n865 = n861 *** (one --- n864);
secret int n866 = (one --- n861) *** n864;
secret int f60 = (one --- n865) *** (one --- n866);
secret int n868 = (one --- n861) *** (one --- n862);
secret int n869 = (one --- n863) *** (one --- n868);
secret int n870 = (one --- a61) *** (one --- b61);
secret int n871 = a61 *** b61;
secret int n872 = (one --- n870) *** (one --- n871);
secret int n873 = n869 *** (one --- n872);
secret int n874 = (one --- n869) *** n872;
secret int f61 = (one --- n873) *** (one --- n874);
secret int n876 = (one --- n869) *** (one --- n870);
secret int n877 = (one --- n871) *** (one --- n876);
secret int n878 = (one --- a62) *** (one --- b62);
secret int n879 = a62 *** b62;
secret int n880 = (one --- n878) *** (one --- n879);
secret int n881 = n877 *** (one --- n880);
secret int n882 = (one --- n877) *** n880;
secret int f62 = (one --- n881) *** (one --- n882);
secret int n884 = (one --- n877) *** (one --- n878);
secret int n885 = (one --- n879) *** (one --- n884);
secret int n886 = (one --- a63) *** (one --- b63);
secret int n887 = a63 *** b63;
secret int n888 = (one --- n886) *** (one --- n887);
secret int n889 = n885 *** (one --- n888);
secret int n890 = (one --- n885) *** n888;
secret int f63 = (one --- n889) *** (one --- n890);
secret int n892 = (one --- n885) *** (one --- n886);
secret int n893 = (one --- n887) *** (one --- n892);
secret int n894 = (one --- a64) *** (one --- b64);
secret int n895 = a64 *** b64;
secret int n896 = (one --- n894) *** (one --- n895);
secret int n897 = n893 *** (one --- n896);
secret int n898 = (one --- n893) *** n896;
secret int f64 = (one --- n897) *** (one --- n898);
secret int n900 = (one --- n893) *** (one --- n894);
secret int n901 = (one --- n895) *** (one --- n900);
secret int n902 = (one --- a65) *** (one --- b65);
secret int n903 = a65 *** b65;
secret int n904 = (one --- n902) *** (one --- n903);
secret int n905 = n901 *** (one --- n904);
secret int n906 = (one --- n901) *** n904;
secret int f65 = (one --- n905) *** (one --- n906);
secret int n908 = (one --- n901) *** (one --- n902);
secret int n909 = (one --- n903) *** (one --- n908);
secret int n910 = (one --- a66) *** (one --- b66);
secret int n911 = a66 *** b66;
secret int n912 = (one --- n910) *** (one --- n911);
secret int n913 = n909 *** (one --- n912);
secret int n914 = (one --- n909) *** n912;
secret int f66 = (one --- n913) *** (one --- n914);
secret int n916 = (one --- n909) *** (one --- n910);
secret int n917 = (one --- n911) *** (one --- n916);
secret int n918 = (one --- a67) *** (one --- b67);
secret int n919 = a67 *** b67;
secret int n920 = (one --- n918) *** (one --- n919);
secret int n921 = n917 *** (one --- n920);
secret int n922 = (one --- n917) *** n920;
secret int f67 = (one --- n921) *** (one --- n922);
secret int n924 = (one --- n917) *** (one --- n918);
secret int n925 = (one --- n919) *** (one --- n924);
secret int n926 = (one --- a68) *** (one --- b68);
secret int n927 = a68 *** b68;
secret int n928 = (one --- n926) *** (one --- n927);
secret int n929 = n925 *** (one --- n928);
secret int n930 = (one --- n925) *** n928;
secret int f68 = (one --- n929) *** (one --- n930);
secret int n932 = (one --- n925) *** (one --- n926);
secret int n933 = (one --- n927) *** (one --- n932);
secret int n934 = (one --- a69) *** (one --- b69);
secret int n935 = a69 *** b69;
secret int n936 = (one --- n934) *** (one --- n935);
secret int n937 = n933 *** (one --- n936);
secret int n938 = (one --- n933) *** n936;
secret int f69 = (one --- n937) *** (one --- n938);
secret int n940 = (one --- n933) *** (one --- n934);
secret int n941 = (one --- n935) *** (one --- n940);
secret int n942 = (one --- a70) *** (one --- b70);
secret int n943 = a70 *** b70;
secret int n944 = (one --- n942) *** (one --- n943);
secret int n945 = n941 *** (one --- n944);
secret int n946 = (one --- n941) *** n944;
secret int f70 = (one --- n945) *** (one --- n946);
secret int n948 = (one --- n941) *** (one --- n942);
secret int n949 = (one --- n943) *** (one --- n948);
secret int n950 = (one --- a71) *** (one --- b71);
secret int n951 = a71 *** b71;
secret int n952 = (one --- n950) *** (one --- n951);
secret int n953 = n949 *** (one --- n952);
secret int n954 = (one --- n949) *** n952;
secret int f71 = (one --- n953) *** (one --- n954);
secret int n956 = (one --- n949) *** (one --- n950);
secret int n957 = (one --- n951) *** (one --- n956);
secret int n958 = (one --- a72) *** (one --- b72);
secret int n959 = a72 *** b72;
secret int n960 = (one --- n958) *** (one --- n959);
secret int n961 = n957 *** (one --- n960);
secret int n962 = (one --- n957) *** n960;
secret int f72 = (one --- n961) *** (one --- n962);
secret int n964 = (one --- n957) *** (one --- n958);
secret int n965 = (one --- n959) *** (one --- n964);
secret int n966 = (one --- a73) *** (one --- b73);
secret int n967 = a73 *** b73;
secret int n968 = (one --- n966) *** (one --- n967);
secret int n969 = n965 *** (one --- n968);
secret int n970 = (one --- n965) *** n968;
secret int f73 = (one --- n969) *** (one --- n970);
secret int n972 = (one --- n965) *** (one --- n966);
secret int n973 = (one --- n967) *** (one --- n972);
secret int n974 = (one --- a74) *** (one --- b74);
secret int n975 = a74 *** b74;
secret int n976 = (one --- n974) *** (one --- n975);
secret int n977 = n973 *** (one --- n976);
secret int n978 = (one --- n973) *** n976;
secret int f74 = (one --- n977) *** (one --- n978);
secret int n980 = (one --- n973) *** (one --- n974);
secret int n981 = (one --- n975) *** (one --- n980);
secret int n982 = (one --- a75) *** (one --- b75);
secret int n983 = a75 *** b75;
secret int n984 = (one --- n982) *** (one --- n983);
secret int n985 = n981 *** (one --- n984);
secret int n986 = (one --- n981) *** n984;
secret int f75 = (one --- n985) *** (one --- n986);
secret int n988 = (one --- n981) *** (one --- n982);
secret int n989 = (one --- n983) *** (one --- n988);
secret int n990 = (one --- a76) *** (one --- b76);
secret int n991 = a76 *** b76;
secret int n992 = (one --- n990) *** (one --- n991);
secret int n993 = n989 *** (one --- n992);
secret int n994 = (one --- n989) *** n992;
secret int f76 = (one --- n993) *** (one --- n994);
secret int n996 = (one --- n989) *** (one --- n990);
secret int n997 = (one --- n991) *** (one --- n996);
secret int n998 = (one --- a77) *** (one --- b77);
secret int n999 = a77 *** b77;
secret int n1000 = (one --- n998) *** (one --- n999);
secret int n1001 = n997 *** (one --- n1000);
secret int n1002 = (one --- n997) *** n1000;
secret int f77 = (one --- n1001) *** (one --- n1002);
secret int n1004 = (one --- n997) *** (one --- n998);
secret int n1005 = (one --- n999) *** (one --- n1004);
secret int n1006 = (one --- a78) *** (one --- b78);
secret int n1007 = a78 *** b78;
secret int n1008 = (one --- n1006) *** (one --- n1007);
secret int n1009 = n1005 *** (one --- n1008);
secret int n1010 = (one --- n1005) *** n1008;
secret int f78 = (one --- n1009) *** (one --- n1010);
secret int n1012 = (one --- n1005) *** (one --- n1006);
secret int n1013 = (one --- n1007) *** (one --- n1012);
secret int n1014 = (one --- a79) *** (one --- b79);
secret int n1015 = a79 *** b79;
secret int n1016 = (one --- n1014) *** (one --- n1015);
secret int n1017 = n1013 *** (one --- n1016);
secret int n1018 = (one --- n1013) *** n1016;
secret int f79 = (one --- n1017) *** (one --- n1018);
secret int n1020 = (one --- n1013) *** (one --- n1014);
secret int n1021 = (one --- n1015) *** (one --- n1020);
secret int n1022 = (one --- a80) *** (one --- b80);
secret int n1023 = a80 *** b80;
secret int n1024 = (one --- n1022) *** (one --- n1023);
secret int n1025 = n1021 *** (one --- n1024);
secret int n1026 = (one --- n1021) *** n1024;
secret int f80 = (one --- n1025) *** (one --- n1026);
secret int n1028 = (one --- n1021) *** (one --- n1022);
secret int n1029 = (one --- n1023) *** (one --- n1028);
secret int n1030 = (one --- a81) *** (one --- b81);
secret int n1031 = a81 *** b81;
secret int n1032 = (one --- n1030) *** (one --- n1031);
secret int n1033 = n1029 *** (one --- n1032);
secret int n1034 = (one --- n1029) *** n1032;
secret int f81 = (one --- n1033) *** (one --- n1034);
secret int n1036 = (one --- n1029) *** (one --- n1030);
secret int n1037 = (one --- n1031) *** (one --- n1036);
secret int n1038 = (one --- a82) *** (one --- b82);
secret int n1039 = a82 *** b82;
secret int n1040 = (one --- n1038) *** (one --- n1039);
secret int n1041 = n1037 *** (one --- n1040);
secret int n1042 = (one --- n1037) *** n1040;
secret int f82 = (one --- n1041) *** (one --- n1042);
secret int n1044 = (one --- n1037) *** (one --- n1038);
secret int n1045 = (one --- n1039) *** (one --- n1044);
secret int n1046 = (one --- a83) *** (one --- b83);
secret int n1047 = a83 *** b83;
secret int n1048 = (one --- n1046) *** (one --- n1047);
secret int n1049 = n1045 *** (one --- n1048);
secret int n1050 = (one --- n1045) *** n1048;
secret int f83 = (one --- n1049) *** (one --- n1050);
secret int n1052 = (one --- n1045) *** (one --- n1046);
secret int n1053 = (one --- n1047) *** (one --- n1052);
secret int n1054 = (one --- a84) *** (one --- b84);
secret int n1055 = a84 *** b84;
secret int n1056 = (one --- n1054) *** (one --- n1055);
secret int n1057 = n1053 *** (one --- n1056);
secret int n1058 = (one --- n1053) *** n1056;
secret int f84 = (one --- n1057) *** (one --- n1058);
secret int n1060 = (one --- n1053) *** (one --- n1054);
secret int n1061 = (one --- n1055) *** (one --- n1060);
secret int n1062 = (one --- a85) *** (one --- b85);
secret int n1063 = a85 *** b85;
secret int n1064 = (one --- n1062) *** (one --- n1063);
secret int n1065 = n1061 *** (one --- n1064);
secret int n1066 = (one --- n1061) *** n1064;
secret int f85 = (one --- n1065) *** (one --- n1066);
secret int n1068 = (one --- n1061) *** (one --- n1062);
secret int n1069 = (one --- n1063) *** (one --- n1068);
secret int n1070 = (one --- a86) *** (one --- b86);
secret int n1071 = a86 *** b86;
secret int n1072 = (one --- n1070) *** (one --- n1071);
secret int n1073 = n1069 *** (one --- n1072);
secret int n1074 = (one --- n1069) *** n1072;
secret int f86 = (one --- n1073) *** (one --- n1074);
secret int n1076 = (one --- n1069) *** (one --- n1070);
secret int n1077 = (one --- n1071) *** (one --- n1076);
secret int n1078 = (one --- a87) *** (one --- b87);
secret int n1079 = a87 *** b87;
secret int n1080 = (one --- n1078) *** (one --- n1079);
secret int n1081 = n1077 *** (one --- n1080);
secret int n1082 = (one --- n1077) *** n1080;
secret int f87 = (one --- n1081) *** (one --- n1082);
secret int n1084 = (one --- n1077) *** (one --- n1078);
secret int n1085 = (one --- n1079) *** (one --- n1084);
secret int n1086 = (one --- a88) *** (one --- b88);
secret int n1087 = a88 *** b88;
secret int n1088 = (one --- n1086) *** (one --- n1087);
secret int n1089 = n1085 *** (one --- n1088);
secret int n1090 = (one --- n1085) *** n1088;
secret int f88 = (one --- n1089) *** (one --- n1090);
secret int n1092 = (one --- n1085) *** (one --- n1086);
secret int n1093 = (one --- n1087) *** (one --- n1092);
secret int n1094 = (one --- a89) *** (one --- b89);
secret int n1095 = a89 *** b89;
secret int n1096 = (one --- n1094) *** (one --- n1095);
secret int n1097 = n1093 *** (one --- n1096);
secret int n1098 = (one --- n1093) *** n1096;
secret int f89 = (one --- n1097) *** (one --- n1098);
secret int n1100 = (one --- n1093) *** (one --- n1094);
secret int n1101 = (one --- n1095) *** (one --- n1100);
secret int n1102 = (one --- a90) *** (one --- b90);
secret int n1103 = a90 *** b90;
secret int n1104 = (one --- n1102) *** (one --- n1103);
secret int n1105 = n1101 *** (one --- n1104);
secret int n1106 = (one --- n1101) *** n1104;
secret int f90 = (one --- n1105) *** (one --- n1106);
secret int n1108 = (one --- n1101) *** (one --- n1102);
secret int n1109 = (one --- n1103) *** (one --- n1108);
secret int n1110 = (one --- a91) *** (one --- b91);
secret int n1111 = a91 *** b91;
secret int n1112 = (one --- n1110) *** (one --- n1111);
secret int n1113 = n1109 *** (one --- n1112);
secret int n1114 = (one --- n1109) *** n1112;
secret int f91 = (one --- n1113) *** (one --- n1114);
secret int n1116 = (one --- n1109) *** (one --- n1110);
secret int n1117 = (one --- n1111) *** (one --- n1116);
secret int n1118 = (one --- a92) *** (one --- b92);
secret int n1119 = a92 *** b92;
secret int n1120 = (one --- n1118) *** (one --- n1119);
secret int n1121 = n1117 *** (one --- n1120);
secret int n1122 = (one --- n1117) *** n1120;
secret int f92 = (one --- n1121) *** (one --- n1122);
secret int n1124 = (one --- n1117) *** (one --- n1118);
secret int n1125 = (one --- n1119) *** (one --- n1124);
secret int n1126 = (one --- a93) *** (one --- b93);
secret int n1127 = a93 *** b93;
secret int n1128 = (one --- n1126) *** (one --- n1127);
secret int n1129 = n1125 *** (one --- n1128);
secret int n1130 = (one --- n1125) *** n1128;
secret int f93 = (one --- n1129) *** (one --- n1130);
secret int n1132 = (one --- n1125) *** (one --- n1126);
secret int n1133 = (one --- n1127) *** (one --- n1132);
secret int n1134 = (one --- a94) *** (one --- b94);
secret int n1135 = a94 *** b94;
secret int n1136 = (one --- n1134) *** (one --- n1135);
secret int n1137 = n1133 *** (one --- n1136);
secret int n1138 = (one --- n1133) *** n1136;
secret int f94 = (one --- n1137) *** (one --- n1138);
secret int n1140 = (one --- n1133) *** (one --- n1134);
secret int n1141 = (one --- n1135) *** (one --- n1140);
secret int n1142 = (one --- a95) *** (one --- b95);
secret int n1143 = a95 *** b95;
secret int n1144 = (one --- n1142) *** (one --- n1143);
secret int n1145 = n1141 *** (one --- n1144);
secret int n1146 = (one --- n1141) *** n1144;
secret int f95 = (one --- n1145) *** (one --- n1146);
secret int n1148 = (one --- n1141) *** (one --- n1142);
secret int n1149 = (one --- n1143) *** (one --- n1148);
secret int n1150 = (one --- a96) *** (one --- b96);
secret int n1151 = a96 *** b96;
secret int n1152 = (one --- n1150) *** (one --- n1151);
secret int n1153 = n1149 *** (one --- n1152);
secret int n1154 = (one --- n1149) *** n1152;
secret int f96 = (one --- n1153) *** (one --- n1154);
secret int n1156 = (one --- n1149) *** (one --- n1150);
secret int n1157 = (one --- n1151) *** (one --- n1156);
secret int n1158 = (one --- a97) *** (one --- b97);
secret int n1159 = a97 *** b97;
secret int n1160 = (one --- n1158) *** (one --- n1159);
secret int n1161 = n1157 *** (one --- n1160);
secret int n1162 = (one --- n1157) *** n1160;
secret int f97 = (one --- n1161) *** (one --- n1162);
secret int n1164 = (one --- n1157) *** (one --- n1158);
secret int n1165 = (one --- n1159) *** (one --- n1164);
secret int n1166 = (one --- a98) *** (one --- b98);
secret int n1167 = a98 *** b98;
secret int n1168 = (one --- n1166) *** (one --- n1167);
secret int n1169 = n1165 *** (one --- n1168);
secret int n1170 = (one --- n1165) *** n1168;
secret int f98 = (one --- n1169) *** (one --- n1170);
secret int n1172 = (one --- n1165) *** (one --- n1166);
secret int n1173 = (one --- n1167) *** (one --- n1172);
secret int n1174 = (one --- a99) *** (one --- b99);
secret int n1175 = a99 *** b99;
secret int n1176 = (one --- n1174) *** (one --- n1175);
secret int n1177 = n1173 *** (one --- n1176);
secret int n1178 = (one --- n1173) *** n1176;
secret int f99 = (one --- n1177) *** (one --- n1178);
secret int n1180 = (one --- n1173) *** (one --- n1174);
secret int n1181 = (one --- n1175) *** (one --- n1180);
secret int n1182 = (one --- a100) *** (one --- b100);
secret int n1183 = a100 *** b100;
secret int n1184 = (one --- n1182) *** (one --- n1183);
secret int n1185 = n1181 *** (one --- n1184);
secret int n1186 = (one --- n1181) *** n1184;
secret int f100 = (one --- n1185) *** (one --- n1186);
secret int n1188 = (one --- n1181) *** (one --- n1182);
secret int n1189 = (one --- n1183) *** (one --- n1188);
secret int n1190 = (one --- a101) *** (one --- b101);
secret int n1191 = a101 *** b101;
secret int n1192 = (one --- n1190) *** (one --- n1191);
secret int n1193 = n1189 *** (one --- n1192);
secret int n1194 = (one --- n1189) *** n1192;
secret int f101 = (one --- n1193) *** (one --- n1194);
secret int n1196 = (one --- n1189) *** (one --- n1190);
secret int n1197 = (one --- n1191) *** (one --- n1196);
secret int n1198 = (one --- a102) *** (one --- b102);
secret int n1199 = a102 *** b102;
secret int n1200 = (one --- n1198) *** (one --- n1199);
secret int n1201 = n1197 *** (one --- n1200);
secret int n1202 = (one --- n1197) *** n1200;
secret int f102 = (one --- n1201) *** (one --- n1202);
secret int n1204 = (one --- n1197) *** (one --- n1198);
secret int n1205 = (one --- n1199) *** (one --- n1204);
secret int n1206 = (one --- a103) *** (one --- b103);
secret int n1207 = a103 *** b103;
secret int n1208 = (one --- n1206) *** (one --- n1207);
secret int n1209 = n1205 *** (one --- n1208);
secret int n1210 = (one --- n1205) *** n1208;
secret int f103 = (one --- n1209) *** (one --- n1210);
secret int n1212 = (one --- n1205) *** (one --- n1206);
secret int n1213 = (one --- n1207) *** (one --- n1212);
secret int n1214 = (one --- a104) *** (one --- b104);
secret int n1215 = a104 *** b104;
secret int n1216 = (one --- n1214) *** (one --- n1215);
secret int n1217 = n1213 *** (one --- n1216);
secret int n1218 = (one --- n1213) *** n1216;
secret int f104 = (one --- n1217) *** (one --- n1218);
secret int n1220 = (one --- n1213) *** (one --- n1214);
secret int n1221 = (one --- n1215) *** (one --- n1220);
secret int n1222 = (one --- a105) *** (one --- b105);
secret int n1223 = a105 *** b105;
secret int n1224 = (one --- n1222) *** (one --- n1223);
secret int n1225 = n1221 *** (one --- n1224);
secret int n1226 = (one --- n1221) *** n1224;
secret int f105 = (one --- n1225) *** (one --- n1226);
secret int n1228 = (one --- n1221) *** (one --- n1222);
secret int n1229 = (one --- n1223) *** (one --- n1228);
secret int n1230 = (one --- a106) *** (one --- b106);
secret int n1231 = a106 *** b106;
secret int n1232 = (one --- n1230) *** (one --- n1231);
secret int n1233 = n1229 *** (one --- n1232);
secret int n1234 = (one --- n1229) *** n1232;
secret int f106 = (one --- n1233) *** (one --- n1234);
secret int n1236 = (one --- n1229) *** (one --- n1230);
secret int n1237 = (one --- n1231) *** (one --- n1236);
secret int n1238 = (one --- a107) *** (one --- b107);
secret int n1239 = a107 *** b107;
secret int n1240 = (one --- n1238) *** (one --- n1239);
secret int n1241 = n1237 *** (one --- n1240);
secret int n1242 = (one --- n1237) *** n1240;
secret int f107 = (one --- n1241) *** (one --- n1242);
secret int n1244 = (one --- n1237) *** (one --- n1238);
secret int n1245 = (one --- n1239) *** (one --- n1244);
secret int n1246 = (one --- a108) *** (one --- b108);
secret int n1247 = a108 *** b108;
secret int n1248 = (one --- n1246) *** (one --- n1247);
secret int n1249 = n1245 *** (one --- n1248);
secret int n1250 = (one --- n1245) *** n1248;
secret int f108 = (one --- n1249) *** (one --- n1250);
secret int n1252 = (one --- n1245) *** (one --- n1246);
secret int n1253 = (one --- n1247) *** (one --- n1252);
secret int n1254 = (one --- a109) *** (one --- b109);
secret int n1255 = a109 *** b109;
secret int n1256 = (one --- n1254) *** (one --- n1255);
secret int n1257 = n1253 *** (one --- n1256);
secret int n1258 = (one --- n1253) *** n1256;
secret int f109 = (one --- n1257) *** (one --- n1258);
secret int n1260 = (one --- n1253) *** (one --- n1254);
secret int n1261 = (one --- n1255) *** (one --- n1260);
secret int n1262 = (one --- a110) *** (one --- b110);
secret int n1263 = a110 *** b110;
secret int n1264 = (one --- n1262) *** (one --- n1263);
secret int n1265 = n1261 *** (one --- n1264);
secret int n1266 = (one --- n1261) *** n1264;
secret int f110 = (one --- n1265) *** (one --- n1266);
secret int n1268 = (one --- n1261) *** (one --- n1262);
secret int n1269 = (one --- n1263) *** (one --- n1268);
secret int n1270 = (one --- a111) *** (one --- b111);
secret int n1271 = a111 *** b111;
secret int n1272 = (one --- n1270) *** (one --- n1271);
secret int n1273 = n1269 *** (one --- n1272);
secret int n1274 = (one --- n1269) *** n1272;
secret int f111 = (one --- n1273) *** (one --- n1274);
secret int n1276 = (one --- n1269) *** (one --- n1270);
secret int n1277 = (one --- n1271) *** (one --- n1276);
secret int n1278 = (one --- a112) *** (one --- b112);
secret int n1279 = a112 *** b112;
secret int n1280 = (one --- n1278) *** (one --- n1279);
secret int n1281 = n1277 *** (one --- n1280);
secret int n1282 = (one --- n1277) *** n1280;
secret int f112 = (one --- n1281) *** (one --- n1282);
secret int n1284 = (one --- n1277) *** (one --- n1278);
secret int n1285 = (one --- n1279) *** (one --- n1284);
secret int n1286 = (one --- a113) *** (one --- b113);
secret int n1287 = a113 *** b113;
secret int n1288 = (one --- n1286) *** (one --- n1287);
secret int n1289 = n1285 *** (one --- n1288);
secret int n1290 = (one --- n1285) *** n1288;
secret int f113 = (one --- n1289) *** (one --- n1290);
secret int n1292 = (one --- n1285) *** (one --- n1286);
secret int n1293 = (one --- n1287) *** (one --- n1292);
secret int n1294 = (one --- a114) *** (one --- b114);
secret int n1295 = a114 *** b114;
secret int n1296 = (one --- n1294) *** (one --- n1295);
secret int n1297 = n1293 *** (one --- n1296);
secret int n1298 = (one --- n1293) *** n1296;
secret int f114 = (one --- n1297) *** (one --- n1298);
secret int n1300 = (one --- n1293) *** (one --- n1294);
secret int n1301 = (one --- n1295) *** (one --- n1300);
secret int n1302 = (one --- a115) *** (one --- b115);
secret int n1303 = a115 *** b115;
secret int n1304 = (one --- n1302) *** (one --- n1303);
secret int n1305 = n1301 *** (one --- n1304);
secret int n1306 = (one --- n1301) *** n1304;
secret int f115 = (one --- n1305) *** (one --- n1306);
secret int n1308 = (one --- n1301) *** (one --- n1302);
secret int n1309 = (one --- n1303) *** (one --- n1308);
secret int n1310 = (one --- a116) *** (one --- b116);
secret int n1311 = a116 *** b116;
secret int n1312 = (one --- n1310) *** (one --- n1311);
secret int n1313 = n1309 *** (one --- n1312);
secret int n1314 = (one --- n1309) *** n1312;
secret int f116 = (one --- n1313) *** (one --- n1314);
secret int n1316 = (one --- n1309) *** (one --- n1310);
secret int n1317 = (one --- n1311) *** (one --- n1316);
secret int n1318 = (one --- a117) *** (one --- b117);
secret int n1319 = a117 *** b117;
secret int n1320 = (one --- n1318) *** (one --- n1319);
secret int n1321 = n1317 *** (one --- n1320);
secret int n1322 = (one --- n1317) *** n1320;
secret int f117 = (one --- n1321) *** (one --- n1322);
secret int n1324 = (one --- n1317) *** (one --- n1318);
secret int n1325 = (one --- n1319) *** (one --- n1324);
secret int n1326 = (one --- a118) *** (one --- b118);
secret int n1327 = a118 *** b118;
secret int n1328 = (one --- n1326) *** (one --- n1327);
secret int n1329 = n1325 *** (one --- n1328);
secret int n1330 = (one --- n1325) *** n1328;
secret int f118 = (one --- n1329) *** (one --- n1330);
secret int n1332 = (one --- n1325) *** (one --- n1326);
secret int n1333 = (one --- n1327) *** (one --- n1332);
secret int n1334 = (one --- a119) *** (one --- b119);
secret int n1335 = a119 *** b119;
secret int n1336 = (one --- n1334) *** (one --- n1335);
secret int n1337 = n1333 *** (one --- n1336);
secret int n1338 = (one --- n1333) *** n1336;
secret int f119 = (one --- n1337) *** (one --- n1338);
secret int n1340 = (one --- n1333) *** (one --- n1334);
secret int n1341 = (one --- n1335) *** (one --- n1340);
secret int n1342 = (one --- a120) *** (one --- b120);
secret int n1343 = a120 *** b120;
secret int n1344 = (one --- n1342) *** (one --- n1343);
secret int n1345 = n1341 *** (one --- n1344);
secret int n1346 = (one --- n1341) *** n1344;
secret int f120 = (one --- n1345) *** (one --- n1346);
secret int n1348 = (one --- n1341) *** (one --- n1342);
secret int n1349 = (one --- n1343) *** (one --- n1348);
secret int n1350 = (one --- a121) *** (one --- b121);
secret int n1351 = a121 *** b121;
secret int n1352 = (one --- n1350) *** (one --- n1351);
secret int n1353 = n1349 *** (one --- n1352);
secret int n1354 = (one --- n1349) *** n1352;
secret int f121 = (one --- n1353) *** (one --- n1354);
secret int n1356 = (one --- n1349) *** (one --- n1350);
secret int n1357 = (one --- n1351) *** (one --- n1356);
secret int n1358 = (one --- a122) *** (one --- b122);
secret int n1359 = a122 *** b122;
secret int n1360 = (one --- n1358) *** (one --- n1359);
secret int n1361 = n1357 *** (one --- n1360);
secret int n1362 = (one --- n1357) *** n1360;
secret int f122 = (one --- n1361) *** (one --- n1362);
secret int n1364 = (one --- n1357) *** (one --- n1358);
secret int n1365 = (one --- n1359) *** (one --- n1364);
secret int n1366 = (one --- a123) *** (one --- b123);
secret int n1367 = a123 *** b123;
secret int n1368 = (one --- n1366) *** (one --- n1367);
secret int n1369 = n1365 *** (one --- n1368);
secret int n1370 = (one --- n1365) *** n1368;
secret int f123 = (one --- n1369) *** (one --- n1370);
secret int n1372 = (one --- n1365) *** (one --- n1366);
secret int n1373 = (one --- n1367) *** (one --- n1372);
secret int n1374 = (one --- a124) *** (one --- b124);
secret int n1375 = a124 *** b124;
secret int n1376 = (one --- n1374) *** (one --- n1375);
secret int n1377 = n1373 *** (one --- n1376);
secret int n1378 = (one --- n1373) *** n1376;
secret int f124 = (one --- n1377) *** (one --- n1378);
secret int n1380 = (one --- n1373) *** (one --- n1374);
secret int n1381 = (one --- n1375) *** (one --- n1380);
secret int n1382 = (one --- a125) *** (one --- b125);
secret int n1383 = a125 *** b125;
secret int n1384 = (one --- n1382) *** (one --- n1383);
secret int n1385 = n1381 *** (one --- n1384);
secret int n1386 = (one --- n1381) *** n1384;
secret int f125 = (one --- n1385) *** (one --- n1386);
secret int n1388 = (one --- n1381) *** (one --- n1382);
secret int n1389 = (one --- n1383) *** (one --- n1388);
secret int n1390 = (one --- a126) *** (one --- b126);
secret int n1391 = a126 *** b126;
secret int n1392 = (one --- n1390) *** (one --- n1391);
secret int n1393 = n1389 *** (one --- n1392);
secret int n1394 = (one --- n1389) *** n1392;
secret int f126 = (one --- n1393) *** (one --- n1394);
secret int n1396 = (one --- n1389) *** (one --- n1390);
secret int n1397 = (one --- n1391) *** (one --- n1396);
secret int n1398 = (one --- a127) *** (one --- b127);
secret int n1399 = a127 *** b127;
secret int n1400 = (one --- n1398) *** (one --- n1399);
secret int n1401 = n1397 *** (one --- n1400);
secret int n1402 = (one --- n1397) *** n1400;
secret int f127 = (one --- n1401) *** (one --- n1402);
secret int n1404 = (one --- n1397) *** (one --- n1398);
secret int cOut = (n1399 +++ n1404 --- n1399 *** n1404);
    )"""";
  auto astProgram = Parser::parse(std::string(program));

  // program's output
  const char *outputs = R""""(
      y = cOut;
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
  registerInputVariable(*rootScope, "b0", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b1", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b2", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b3", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b4", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b5", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b6", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b7", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b8", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b9", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b10", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b11", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b12", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b13", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b14", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b15", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b16", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b17", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b18", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b19", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b20", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b21", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b22", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b23", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b24", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b25", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b26", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b27", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b28", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b29", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b30", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b31", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b32", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b33", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b34", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b35", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b36", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b37", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b38", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b39", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b40", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b41", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b42", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b43", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b44", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b45", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b46", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b47", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b48", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b49", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b50", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b51", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b52", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b53", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b54", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b55", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b56", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b57", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b58", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b59", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b60", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b61", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b62", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b63", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b64", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b65", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b66", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b67", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b68", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b69", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b70", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b71", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b72", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b73", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b74", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b75", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b76", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b77", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b78", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b79", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b80", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b81", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b82", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b83", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b84", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b85", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b86", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b87", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b88", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b89", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b90", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b91", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b92", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b93", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b94", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b95", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b96", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b97", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b98", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b99", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b100", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b101", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b102", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b103", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b104", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b105", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b106", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b107", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b108", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b109", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b110", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b111", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b112", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b113", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b114", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b115", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b116", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b117", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b118", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b119", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b120", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b121", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b122", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b123", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b124", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b125", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b126", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b127", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f0", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f1", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f2", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f3", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f4", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f5", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f6", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f7", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f8", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f9", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f10", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f11", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f12", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f13", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f14", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f15", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f16", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f17", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f18", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f19", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f20", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f21", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f22", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f23", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f24", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f25", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f26", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f27", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f28", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f29", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f30", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f31", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f32", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f33", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f34", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f35", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f36", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f37", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f38", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f39", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f40", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f41", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f42", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f43", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f44", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f45", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f46", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f47", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f48", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f49", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f50", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f51", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f52", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f53", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f54", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f55", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f56", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f57", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f58", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f59", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f60", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f61", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f62", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f63", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f64", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f65", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f66", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f67", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f68", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f69", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f70", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f71", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f72", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f73", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f74", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f75", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f76", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f77", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f78", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f79", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f80", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f81", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f82", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f83", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f84", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f85", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f86", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f87", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f88", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f89", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f90", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f91", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f92", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f93", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f94", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f95", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f96", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f97", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f98", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f99", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f100", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f101", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f102", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f103", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f104", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f105", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f106", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f107", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f108", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f109", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f110", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f111", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f112", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f113", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f114", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f115", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f116", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f117", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f118", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f119", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f120", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f121", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f122", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f123", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f124", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f125", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f126", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f127", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "one", Datatype(Type::INT, true));

  std::stringstream rs;
  ProgramPrintVisitor p(rs);
  astProgram->accept(p);
  std::cout << rs.str() << std::endl;

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
  }


  // Print Noise Map
  std::stringstream tt;
  NoisePrintVisitor v(tt, srv.getNoiseMap(), srv.getRelNoiseMap());

  astProgram->accept(v);

  std::cout << "Program: " << std::endl;
  std::cout << tt.str() << std::endl;


  std::stringstream ss;
  InsertModSwitchVisitor modSwitchVis(ss, srv.getNoiseMap(), coeffmodulusmap, coeffmodulusmap_vars, calcInitNoiseHeuristic());

  std::cout << "Looking for modswitch insertion sites" << std::endl;

  astProgram->accept(modSwitchVis); // find modswitching nodes

  if (!modSwitchVis.getModSwitchNodes().empty()) {
    std::cout << "Found " << modSwitchVis.getModSwitchNodes().size() << " potential modswitch sites" << std::endl;

    auto binExprIns = modSwitchVis.getModSwitchNodes(); //  modSwitches to be inserted

    auto rewritten_ast = modSwitchVis.insertModSwitchInAst(&astProgram, binExprIns[1], coeffmodulusmap);

    // update noise map
   // modSwitchVis.updateNoiseMap(*rewritten_ast, &srv);

    //update coeff modulus map
    modSwitchVis.updateCoeffModulusMap(binExprIns[1], 1);
    coeffmodulusmap = modSwitchVis.getCoeffModulusMap();

    rewritten_ast = modSwitchVis.insertModSwitchInAst(&rewritten_ast, binExprIns[2], coeffmodulusmap);



//
//    std::cout << "NEW: ";
//    for (auto n : vis.v) {
//      std::cout << coeffmodulusmap[n->getUniqueNodeId()].size() << " ";
//    }

    // print output program
    std::stringstream rr;
    ProgramPrintVisitor p1(rr);
    rewritten_ast->accept(p1);
    std::cout << rr.str() << std::endl;
  } else {
    std::cout << "No ModSwitch Sites found" << std::endl;
  }
}

TEST_F(InsertModSwitchVisitorTest, Adder_rewrite_AST) {
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
secret int b0 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b1 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b2 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b3 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b4 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b5 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b6 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b7 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b8 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b9 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b10 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b11 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b12 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b13 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b14 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b15 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b16 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b17 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b18 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b19 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b20 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b21 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b22 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b23 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b24 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b25 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b26 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b27 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b28 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b29 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b30 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b31 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b32 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b33 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b34 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b35 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b36 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b37 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b38 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b39 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b40 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b41 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b42 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b43 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b44 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b45 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b46 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b47 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b48 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b49 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b50 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b51 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b52 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b53 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b54 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b55 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b56 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b57 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b58 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b59 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b60 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b61 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b62 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b63 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b64 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b65 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b66 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b67 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b68 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b69 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b70 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b71 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b72 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b73 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b74 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b75 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b76 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b77 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b78 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b79 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b80 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b81 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b82 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b83 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b84 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b85 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b86 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b87 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b88 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b89 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b90 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b91 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b92 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b93 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b94 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b95 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b96 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b97 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b98 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b99 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b100 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b101 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b102 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b103 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b104 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b105 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b106 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b107 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b108 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b109 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b110 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b111 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b112 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b113 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b114 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b115 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b116 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b117 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b118 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b119 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b120 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b121 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b122 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b123 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b124 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b125 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b126 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int b127 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f0 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f1 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f2 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f3 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f4 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f5 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f6 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f7 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f8 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f9 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f10 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f11 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f12 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f13 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f14 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f15 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f16 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f17 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f18 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f19 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f20 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f21 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f22 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f23 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f24 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f25 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f26 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f27 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f28 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f29 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f30 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f31 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f32 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f33 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f34 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f35 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f36 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f37 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f38 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f39 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f40 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f41 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f42 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f43 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f44 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f45 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f46 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f47 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f48 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f49 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f50 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f51 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f52 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f53 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f54 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f55 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f56 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f57 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f58 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f59 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f60 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f61 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f62 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f63 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f64 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f65 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f66 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f67 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f68 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f69 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f70 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f71 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f72 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f73 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f74 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f75 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f76 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f77 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f78 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f79 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f80 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f81 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f82 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f83 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f84 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f85 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f86 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f87 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f88 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f89 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f90 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f91 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f92 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f93 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f94 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f95 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f96 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f97 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f98 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f99 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f100 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f101 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f102 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f103 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f104 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f105 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f106 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f107 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f108 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f109 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f110 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f111 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f112 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f113 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f114 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f115 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f116 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f117 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f118 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f119 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f120 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f121 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f122 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f123 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f124 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f125 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f126 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int f127 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int cOut = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
secret int one = {1,  1,   1,   1,  1, 1, 1,  1, 1, 1};
    )"""";
  auto astInput = Parser::parse(std::string(inputs));

  // program specification
  const char *program = R""""(
secret int n386 = a0 *** (one --- b0);
secret int n387 = (one --- a0) *** b0;
secret int f0 = (n386 +++ n387 --- n386 *** n387);
secret int n389 = a0 *** b0;
secret int n390 = (one --- a1) *** (one --- b1);
secret int n391 = a1 *** b1;
secret int n392 = (one --- n390) *** (one --- n391);
secret int n393 = n389 *** (one --- n392);
secret int n394 = (one --- n389) *** n392;
secret int f1 = (n393 +++ n394 --- n393 *** n394);
secret int n396 = n389 *** (one --- n390);
secret int n397 = (one --- n391) *** (one --- n396);
secret int n398 = (one --- a2) *** (one --- b2);
secret int n399 = a2 *** b2;
secret int n400 = (one --- n398) *** (one --- n399);
secret int n401 = n397 *** (one --- n400);
secret int n402 = (one --- n397) *** n400;
secret int f2 = (one --- n401) *** (one --- n402);
secret int n404 = (one --- n397) *** (one --- n398);
secret int n405 = (one --- n399) *** (one --- n404);
secret int n406 = (one --- a3) *** (one --- b3);
secret int n407 = a3 *** b3;
secret int n408 = (one --- n406) *** (one --- n407);
secret int n409 = n405 *** (one --- n408);
secret int n410 = (one --- n405) *** n408;
secret int f3 = (one --- n409) *** (one --- n410);
secret int n412 = (one --- n405) *** (one --- n406);
secret int n413 = (one --- n407) *** (one --- n412);
secret int n414 = (one --- a4) *** (one --- b4);
secret int n415 = a4 *** b4;
secret int n416 = (one --- n414) *** (one --- n415);
secret int n417 = n413 *** (one --- n416);
secret int n418 = (one --- n413) *** n416;
secret int f4 = (one --- n417) *** (one --- n418);
secret int n420 = (one --- n413) *** (one --- n414);
secret int n421 = (one --- n415) *** (one --- n420);
secret int n422 = (one --- a5) *** (one --- b5);
secret int n423 = a5 *** b5;
secret int n424 = (one --- n422) *** (one --- n423);
secret int n425 = n421 *** (one --- n424);
secret int n426 = (one --- n421) *** n424;
secret int f5 = (one --- n425) *** (one --- n426);
secret int n428 = (one --- n421) *** (one --- n422);
secret int n429 = (one --- n423) *** (one --- n428);
secret int n430 = (one --- a6) *** (one --- b6);
secret int n431 = a6 *** b6;
secret int n432 = (one --- n430) *** (one --- n431);
secret int n433 = n429 *** (one --- n432);
secret int n434 = (one --- n429) *** n432;
secret int f6 = (one --- n433) *** (one --- n434);
secret int n436 = (one --- n429) *** (one --- n430);
secret int n437 = (one --- n431) *** (one --- n436);
secret int n438 = (one --- a7) *** (one --- b7);
secret int n439 = a7 *** b7;
secret int n440 = (one --- n438) *** (one --- n439);
secret int n441 = n437 *** (one --- n440);
secret int n442 = (one --- n437) *** n440;
secret int f7 = (one --- n441) *** (one --- n442);
secret int n444 = (one --- n437) *** (one --- n438);
secret int n445 = (one --- n439) *** (one --- n444);
secret int n446 = (one --- a8) *** (one --- b8);
secret int n447 = a8 *** b8;
secret int n448 = (one --- n446) *** (one --- n447);
secret int n449 = n445 *** (one --- n448);
secret int n450 = (one --- n445) *** n448;
secret int f8 = (one --- n449) *** (one --- n450);
secret int n452 = (one --- n445) *** (one --- n446);
secret int n453 = (one --- n447) *** (one --- n452);
secret int n454 = (one --- a9) *** (one --- b9);
secret int n455 = a9 *** b9;
secret int n456 = (one --- n454) *** (one --- n455);
secret int n457 = n453 *** (one --- n456);
secret int n458 = (one --- n453) *** n456;
secret int f9 = (one --- n457) *** (one --- n458);
secret int n460 = (one --- n453) *** (one --- n454);
secret int n461 = (one --- n455) *** (one --- n460);
secret int n462 = (one --- a10) *** (one --- b10);
secret int n463 = a10 *** b10;
secret int n464 = (one --- n462) *** (one --- n463);
secret int n465 = n461 *** (one --- n464);
secret int n466 = (one --- n461) *** n464;
secret int f10 = (one --- n465) *** (one --- n466);
secret int n468 = (one --- n461) *** (one --- n462);
secret int n469 = (one --- n463) *** (one --- n468);
secret int n470 = (one --- a11) *** (one --- b11);
secret int n471 = a11 *** b11;
secret int n472 = (one --- n470) *** (one --- n471);
secret int n473 = n469 *** (one --- n472);
secret int n474 = (one --- n469) *** n472;
secret int f11 = (one --- n473) *** (one --- n474);
secret int n476 = (one --- n469) *** (one --- n470);
secret int n477 = (one --- n471) *** (one --- n476);
secret int n478 = (one --- a12) *** (one --- b12);
secret int n479 = a12 *** b12;
secret int n480 = (one --- n478) *** (one --- n479);
secret int n481 = n477 *** (one --- n480);
secret int n482 = (one --- n477) *** n480;
secret int f12 = (one --- n481) *** (one --- n482);
secret int n484 = (one --- n477) *** (one --- n478);
secret int n485 = (one --- n479) *** (one --- n484);
secret int n486 = (one --- a13) *** (one --- b13);
secret int n487 = a13 *** b13;
secret int n488 = (one --- n486) *** (one --- n487);
secret int n489 = n485 *** (one --- n488);
secret int n490 = (one --- n485) *** n488;
secret int f13 = (one --- n489) *** (one --- n490);
secret int n492 = (one --- n485) *** (one --- n486);
secret int n493 = (one --- n487) *** (one --- n492);
secret int n494 = (one --- a14) *** (one --- b14);
secret int n495 = a14 *** b14;
secret int n496 = (one --- n494) *** (one --- n495);
secret int n497 = n493 *** (one --- n496);
secret int n498 = (one --- n493) *** n496;
secret int f14 = (one --- n497) *** (one --- n498);
secret int n500 = (one --- n493) *** (one --- n494);
secret int n501 = (one --- n495) *** (one --- n500);
secret int n502 = (one --- a15) *** (one --- b15);
secret int n503 = a15 *** b15;
secret int n504 = (one --- n502) *** (one --- n503);
secret int n505 = n501 *** (one --- n504);
secret int n506 = (one --- n501) *** n504;
secret int f15 = (one --- n505) *** (one --- n506);
secret int n508 = (one --- n501) *** (one --- n502);
secret int n509 = (one --- n503) *** (one --- n508);
secret int n510 = (one --- a16) *** (one --- b16);
secret int n511 = a16 *** b16;
secret int n512 = (one --- n510) *** (one --- n511);
secret int n513 = n509 *** (one --- n512);
secret int n514 = (one --- n509) *** n512;
secret int f16 = (one --- n513) *** (one --- n514);
secret int n516 = (one --- n509) *** (one --- n510);
secret int n517 = (one --- n511) *** (one --- n516);
secret int n518 = (one --- a17) *** (one --- b17);
secret int n519 = a17 *** b17;
secret int n520 = (one --- n518) *** (one --- n519);
secret int n521 = n517 *** (one --- n520);
secret int n522 = (one --- n517) *** n520;
secret int f17 = (one --- n521) *** (one --- n522);
secret int n524 = (one --- n517) *** (one --- n518);
secret int n525 = (one --- n519) *** (one --- n524);
secret int n526 = (one --- a18) *** (one --- b18);
secret int n527 = a18 *** b18;
secret int n528 = (one --- n526) *** (one --- n527);
secret int n529 = n525 *** (one --- n528);
secret int n530 = (one --- n525) *** n528;
secret int f18 = (one --- n529) *** (one --- n530);
secret int n532 = (one --- n525) *** (one --- n526);
secret int n533 = (one --- n527) *** (one --- n532);
secret int n534 = (one --- a19) *** (one --- b19);
secret int n535 = a19 *** b19;
secret int n536 = (one --- n534) *** (one --- n535);
secret int n537 = n533 *** (one --- n536);
secret int n538 = (one --- n533) *** n536;
secret int f19 = (one --- n537) *** (one --- n538);
secret int n540 = (one --- n533) *** (one --- n534);
secret int n541 = (one --- n535) *** (one --- n540);
secret int n542 = (one --- a20) *** (one --- b20);
secret int n543 = a20 *** b20;
secret int n544 = (one --- n542) *** (one --- n543);
secret int n545 = n541 *** (one --- n544);
secret int n546 = (one --- n541) *** n544;
secret int f20 = (one --- n545) *** (one --- n546);
secret int n548 = (one --- n541) *** (one --- n542);
secret int n549 = (one --- n543) *** (one --- n548);
secret int n550 = (one --- a21) *** (one --- b21);
secret int n551 = a21 *** b21;
secret int n552 = (one --- n550) *** (one --- n551);
secret int n553 = n549 *** (one --- n552);
secret int n554 = (one --- n549) *** n552;
secret int f21 = (one --- n553) *** (one --- n554);
secret int n556 = (one --- n549) *** (one --- n550);
secret int n557 = (one --- n551) *** (one --- n556);
secret int n558 = (one --- a22) *** (one --- b22);
secret int n559 = a22 *** b22;
secret int n560 = (one --- n558) *** (one --- n559);
secret int n561 = n557 *** (one --- n560);
secret int n562 = (one --- n557) *** n560;
secret int f22 = (one --- n561) *** (one --- n562);
secret int n564 = (one --- n557) *** (one --- n558);
secret int n565 = (one --- n559) *** (one --- n564);
secret int n566 = (one --- a23) *** (one --- b23);
secret int n567 = a23 *** b23;
secret int n568 = (one --- n566) *** (one --- n567);
secret int n569 = n565 *** (one --- n568);
secret int n570 = (one --- n565) *** n568;
secret int f23 = (one --- n569) *** (one --- n570);
secret int n572 = (one --- n565) *** (one --- n566);
secret int n573 = (one --- n567) *** (one --- n572);
secret int n574 = (one --- a24) *** (one --- b24);
secret int n575 = a24 *** b24;
secret int n576 = (one --- n574) *** (one --- n575);
secret int n577 = n573 *** (one --- n576);
secret int n578 = (one --- n573) *** n576;
secret int f24 = (one --- n577) *** (one --- n578);
secret int n580 = (one --- n573) *** (one --- n574);
secret int n581 = (one --- n575) *** (one --- n580);
secret int n582 = (one --- a25) *** (one --- b25);
secret int n583 = a25 *** b25;
secret int n584 = (one --- n582) *** (one --- n583);
secret int n585 = n581 *** (one --- n584);
secret int n586 = (one --- n581) *** n584;
secret int f25 = (one --- n585) *** (one --- n586);
secret int n588 = (one --- n581) *** (one --- n582);
secret int n589 = (one --- n583) *** (one --- n588);
secret int n590 = (one --- a26) *** (one --- b26);
secret int n591 = a26 *** b26;
secret int n592 = (one --- n590) *** (one --- n591);
secret int n593 = n589 *** (one --- n592);
secret int n594 = (one --- n589) *** n592;
secret int f26 = (one --- n593) *** (one --- n594);
secret int n596 = (one --- n589) *** (one --- n590);
secret int n597 = (one --- n591) *** (one --- n596);
secret int n598 = (one --- a27) *** (one --- b27);
secret int n599 = a27 *** b27;
secret int n600 = (one --- n598) *** (one --- n599);
secret int n601 = n597 *** (one --- n600);
secret int n602 = (one --- n597) *** n600;
secret int f27 = (one --- n601) *** (one --- n602);
secret int n604 = (one --- n597) *** (one --- n598);
secret int n605 = (one --- n599) *** (one --- n604);
secret int n606 = (one --- a28) *** (one --- b28);
secret int n607 = a28 *** b28;
secret int n608 = (one --- n606) *** (one --- n607);
secret int n609 = n605 *** (one --- n608);
secret int n610 = (one --- n605) *** n608;
secret int f28 = (one --- n609) *** (one --- n610);
secret int n612 = (one --- n605) *** (one --- n606);
secret int n613 = (one --- n607) *** (one --- n612);
secret int n614 = (one --- a29) *** (one --- b29);
secret int n615 = a29 *** b29;
secret int n616 = (one --- n614) *** (one --- n615);
secret int n617 = n613 *** (one --- n616);
secret int n618 = (one --- n613) *** n616;
secret int f29 = (one --- n617) *** (one --- n618);
secret int n620 = (one --- n613) *** (one --- n614);
secret int n621 = (one --- n615) *** (one --- n620);
secret int n622 = (one --- a30) *** (one --- b30);
secret int n623 = a30 *** b30;
secret int n624 = (one --- n622) *** (one --- n623);
secret int n625 = n621 *** (one --- n624);
secret int n626 = (one --- n621) *** n624;
secret int f30 = (one --- n625) *** (one --- n626);
secret int n628 = (one --- n621) *** (one --- n622);
secret int n629 = (one --- n623) *** (one --- n628);
secret int n630 = (one --- a31) *** (one --- b31);
secret int n631 = a31 *** b31;
secret int n632 = (one --- n630) *** (one --- n631);
secret int n633 = n629 *** (one --- n632);
secret int n634 = (one --- n629) *** n632;
secret int f31 = (one --- n633) *** (one --- n634);
secret int n636 = (one --- n629) *** (one --- n630);
secret int n637 = (one --- n631) *** (one --- n636);
secret int n638 = (one --- a32) *** (one --- b32);
secret int n639 = a32 *** b32;
secret int n640 = (one --- n638) *** (one --- n639);
secret int n641 = n637 *** (one --- n640);
secret int n642 = (one --- n637) *** n640;
secret int f32 = (one --- n641) *** (one --- n642);
secret int n644 = (one --- n637) *** (one --- n638);
secret int n645 = (one --- n639) *** (one --- n644);
secret int n646 = (one --- a33) *** (one --- b33);
secret int n647 = a33 *** b33;
secret int n648 = (one --- n646) *** (one --- n647);
secret int n649 = n645 *** (one --- n648);
secret int n650 = (one --- n645) *** n648;
secret int f33 = (one --- n649) *** (one --- n650);
secret int n652 = (one --- n645) *** (one --- n646);
secret int n653 = (one --- n647) *** (one --- n652);
secret int n654 = (one --- a34) *** (one --- b34);
secret int n655 = a34 *** b34;
secret int n656 = (one --- n654) *** (one --- n655);
secret int n657 = n653 *** (one --- n656);
secret int n658 = (one --- n653) *** n656;
secret int f34 = (one --- n657) *** (one --- n658);
secret int n660 = (one --- n653) *** (one --- n654);
secret int n661 = (one --- n655) *** (one --- n660);
secret int n662 = (one --- a35) *** (one --- b35);
secret int n663 = a35 *** b35;
secret int n664 = (one --- n662) *** (one --- n663);
secret int n665 = n661 *** (one --- n664);
secret int n666 = (one --- n661) *** n664;
secret int f35 = (one --- n665) *** (one --- n666);
secret int n668 = (one --- n661) *** (one --- n662);
secret int n669 = (one --- n663) *** (one --- n668);
secret int n670 = (one --- a36) *** (one --- b36);
secret int n671 = a36 *** b36;
secret int n672 = (one --- n670) *** (one --- n671);
secret int n673 = n669 *** (one --- n672);
secret int n674 = (one --- n669) *** n672;
secret int f36 = (one --- n673) *** (one --- n674);
secret int n676 = (one --- n669) *** (one --- n670);
secret int n677 = (one --- n671) *** (one --- n676);
secret int n678 = (one --- a37) *** (one --- b37);
secret int n679 = a37 *** b37;
secret int n680 = (one --- n678) *** (one --- n679);
secret int n681 = n677 *** (one --- n680);
secret int n682 = (one --- n677) *** n680;
secret int f37 = (one --- n681) *** (one --- n682);
secret int n684 = (one --- n677) *** (one --- n678);
secret int n685 = (one --- n679) *** (one --- n684);
secret int n686 = (one --- a38) *** (one --- b38);
secret int n687 = a38 *** b38;
secret int n688 = (one --- n686) *** (one --- n687);
secret int n689 = n685 *** (one --- n688);
secret int n690 = (one --- n685) *** n688;
secret int f38 = (one --- n689) *** (one --- n690);
secret int n692 = (one --- n685) *** (one --- n686);
secret int n693 = (one --- n687) *** (one --- n692);
secret int n694 = (one --- a39) *** (one --- b39);
secret int n695 = a39 *** b39;
secret int n696 = (one --- n694) *** (one --- n695);
secret int n697 = n693 *** (one --- n696);
secret int n698 = (one --- n693) *** n696;
secret int f39 = (one --- n697) *** (one --- n698);
secret int n700 = (one --- n693) *** (one --- n694);
secret int n701 = (one --- n695) *** (one --- n700);
secret int n702 = (one --- a40) *** (one --- b40);
secret int n703 = a40 *** b40;
secret int n704 = (one --- n702) *** (one --- n703);
secret int n705 = n701 *** (one --- n704);
secret int n706 = (one --- n701) *** n704;
secret int f40 = (one --- n705) *** (one --- n706);
secret int n708 = (one --- n701) *** (one --- n702);
secret int n709 = (one --- n703) *** (one --- n708);
secret int n710 = (one --- a41) *** (one --- b41);
secret int n711 = a41 *** b41;
secret int n712 = (one --- n710) *** (one --- n711);
secret int n713 = n709 *** (one --- n712);
secret int n714 = (one --- n709) *** n712;
secret int f41 = (one --- n713) *** (one --- n714);
secret int n716 = (one --- n709) *** (one --- n710);
secret int n717 = (one --- n711) *** (one --- n716);
secret int n718 = (one --- a42) *** (one --- b42);
secret int n719 = a42 *** b42;
secret int n720 = (one --- n718) *** (one --- n719);
secret int n721 = n717 *** (one --- n720);
secret int n722 = (one --- n717) *** n720;
secret int f42 = (one --- n721) *** (one --- n722);
secret int n724 = (one --- n717) *** (one --- n718);
secret int n725 = (one --- n719) *** (one --- n724);
secret int n726 = (one --- a43) *** (one --- b43);
secret int n727 = a43 *** b43;
secret int n728 = (one --- n726) *** (one --- n727);
secret int n729 = n725 *** (one --- n728);
secret int n730 = (one --- n725) *** n728;
secret int f43 = (one --- n729) *** (one --- n730);
secret int n732 = (one --- n725) *** (one --- n726);
secret int n733 = (one --- n727) *** (one --- n732);
secret int n734 = (one --- a44) *** (one --- b44);
secret int n735 = a44 *** b44;
secret int n736 = (one --- n734) *** (one --- n735);
secret int n737 = n733 *** (one --- n736);
secret int n738 = (one --- n733) *** n736;
secret int f44 = (one --- n737) *** (one --- n738);
secret int n740 = (one --- n733) *** (one --- n734);
secret int n741 = (one --- n735) *** (one --- n740);
secret int n742 = (one --- a45) *** (one --- b45);
secret int n743 = a45 *** b45;
secret int n744 = (one --- n742) *** (one --- n743);
secret int n745 = n741 *** (one --- n744);
secret int n746 = (one --- n741) *** n744;
secret int f45 = (one --- n745) *** (one --- n746);
secret int n748 = (one --- n741) *** (one --- n742);
secret int n749 = (one --- n743) *** (one --- n748);
secret int n750 = (one --- a46) *** (one --- b46);
secret int n751 = a46 *** b46;
secret int n752 = (one --- n750) *** (one --- n751);
secret int n753 = n749 *** (one --- n752);
secret int n754 = (one --- n749) *** n752;
secret int f46 = (one --- n753) *** (one --- n754);
secret int n756 = (one --- n749) *** (one --- n750);
secret int n757 = (one --- n751) *** (one --- n756);
secret int n758 = (one --- a47) *** (one --- b47);
secret int n759 = a47 *** b47;
secret int n760 = (one --- n758) *** (one --- n759);
secret int n761 = n757 *** (one --- n760);
secret int n762 = (one --- n757) *** n760;
secret int f47 = (one --- n761) *** (one --- n762);
secret int n764 = (one --- n757) *** (one --- n758);
secret int n765 = (one --- n759) *** (one --- n764);
secret int n766 = (one --- a48) *** (one --- b48);
secret int n767 = a48 *** b48;
secret int n768 = (one --- n766) *** (one --- n767);
secret int n769 = n765 *** (one --- n768);
secret int n770 = (one --- n765) *** n768;
secret int f48 = (one --- n769) *** (one --- n770);
secret int n772 = (one --- n765) *** (one --- n766);
secret int n773 = (one --- n767) *** (one --- n772);
secret int n774 = (one --- a49) *** (one --- b49);
secret int n775 = a49 *** b49;
secret int n776 = (one --- n774) *** (one --- n775);
secret int n777 = n773 *** (one --- n776);
secret int n778 = (one --- n773) *** n776;
secret int f49 = (one --- n777) *** (one --- n778);
secret int n780 = (one --- n773) *** (one --- n774);
secret int n781 = (one --- n775) *** (one --- n780);
secret int n782 = (one --- a50) *** (one --- b50);
secret int n783 = a50 *** b50;
secret int n784 = (one --- n782) *** (one --- n783);
secret int n785 = n781 *** (one --- n784);
secret int n786 = (one --- n781) *** n784;
secret int f50 = (one --- n785) *** (one --- n786);
secret int n788 = (one --- n781) *** (one --- n782);
secret int n789 = (one --- n783) *** (one --- n788);
secret int n790 = (one --- a51) *** (one --- b51);
secret int n791 = a51 *** b51;
secret int n792 = (one --- n790) *** (one --- n791);
secret int n793 = n789 *** (one --- n792);
secret int n794 = (one --- n789) *** n792;
secret int f51 = (one --- n793) *** (one --- n794);
secret int n796 = (one --- n789) *** (one --- n790);
secret int n797 = (one --- n791) *** (one --- n796);
secret int n798 = (one --- a52) *** (one --- b52);
secret int n799 = a52 *** b52;
secret int n800 = (one --- n798) *** (one --- n799);
secret int n801 = n797 *** (one --- n800);
secret int n802 = (one --- n797) *** n800;
secret int f52 = (one --- n801) *** (one --- n802);
secret int n804 = (one --- n797) *** (one --- n798);
secret int n805 = (one --- n799) *** (one --- n804);
secret int n806 = (one --- a53) *** (one --- b53);
secret int n807 = a53 *** b53;
secret int n808 = (one --- n806) *** (one --- n807);
secret int n809 = n805 *** (one --- n808);
secret int n810 = (one --- n805) *** n808;
secret int f53 = (one --- n809) *** (one --- n810);
secret int n812 = (one --- n805) *** (one --- n806);
secret int n813 = (one --- n807) *** (one --- n812);
secret int n814 = (one --- a54) *** (one --- b54);
secret int n815 = a54 *** b54;
secret int n816 = (one --- n814) *** (one --- n815);
secret int n817 = n813 *** (one --- n816);
secret int n818 = (one --- n813) *** n816;
secret int f54 = (one --- n817) *** (one --- n818);
secret int n820 = (one --- n813) *** (one --- n814);
secret int n821 = (one --- n815) *** (one --- n820);
secret int n822 = (one --- a55) *** (one --- b55);
secret int n823 = a55 *** b55;
secret int n824 = (one --- n822) *** (one --- n823);
secret int n825 = n821 *** (one --- n824);
secret int n826 = (one --- n821) *** n824;
secret int f55 = (one --- n825) *** (one --- n826);
secret int n828 = (one --- n821) *** (one --- n822);
secret int n829 = (one --- n823) *** (one --- n828);
secret int n830 = (one --- a56) *** (one --- b56);
secret int n831 = a56 *** b56;
secret int n832 = (one --- n830) *** (one --- n831);
secret int n833 = n829 *** (one --- n832);
secret int n834 = (one --- n829) *** n832;
secret int f56 = (one --- n833) *** (one --- n834);
secret int n836 = (one --- n829) *** (one --- n830);
secret int n837 = (one --- n831) *** (one --- n836);
secret int n838 = (one --- a57) *** (one --- b57);
secret int n839 = a57 *** b57;
secret int n840 = (one --- n838) *** (one --- n839);
secret int n841 = n837 *** (one --- n840);
secret int n842 = (one --- n837) *** n840;
secret int f57 = (one --- n841) *** (one --- n842);
secret int n844 = (one --- n837) *** (one --- n838);
secret int n845 = (one --- n839) *** (one --- n844);
secret int n846 = (one --- a58) *** (one --- b58);
secret int n847 = a58 *** b58;
secret int n848 = (one --- n846) *** (one --- n847);
secret int n849 = n845 *** (one --- n848);
secret int n850 = (one --- n845) *** n848;
secret int f58 = (one --- n849) *** (one --- n850);
secret int n852 = (one --- n845) *** (one --- n846);
secret int n853 = (one --- n847) *** (one --- n852);
secret int n854 = (one --- a59) *** (one --- b59);
secret int n855 = a59 *** b59;
secret int n856 = (one --- n854) *** (one --- n855);
secret int n857 = n853 *** (one --- n856);
secret int n858 = (one --- n853) *** n856;
secret int f59 = (one --- n857) *** (one --- n858);
secret int n860 = (one --- n853) *** (one --- n854);
secret int n861 = (one --- n855) *** (one --- n860);
secret int n862 = (one --- a60) *** (one --- b60);
secret int n863 = a60 *** b60;
secret int n864 = (one --- n862) *** (one --- n863);
secret int n865 = n861 *** (one --- n864);
secret int n866 = (one --- n861) *** n864;
secret int f60 = (one --- n865) *** (one --- n866);
secret int n868 = (one --- n861) *** (one --- n862);
secret int n869 = (one --- n863) *** (one --- n868);
secret int n870 = (one --- a61) *** (one --- b61);
secret int n871 = a61 *** b61;
secret int n872 = (one --- n870) *** (one --- n871);
secret int n873 = n869 *** (one --- n872);
secret int n874 = (one --- n869) *** n872;
secret int f61 = (one --- n873) *** (one --- n874);
secret int n876 = (one --- n869) *** (one --- n870);
secret int n877 = (one --- n871) *** (one --- n876);
secret int n878 = (one --- a62) *** (one --- b62);
secret int n879 = a62 *** b62;
secret int n880 = (one --- n878) *** (one --- n879);
secret int n881 = n877 *** (one --- n880);
secret int n882 = (one --- n877) *** n880;
secret int f62 = (one --- n881) *** (one --- n882);
secret int n884 = (one --- n877) *** (one --- n878);
secret int n885 = (one --- n879) *** (one --- n884);
secret int n886 = (one --- a63) *** (one --- b63);
secret int n887 = a63 *** b63;
secret int n888 = (one --- n886) *** (one --- n887);
secret int n889 = n885 *** (one --- n888);
secret int n890 = (one --- n885) *** n888;
secret int f63 = (one --- n889) *** (one --- n890);
secret int n892 = (one --- n885) *** (one --- n886);
secret int n893 = (one --- n887) *** (one --- n892);
secret int n894 = (one --- a64) *** (one --- b64);
secret int n895 = a64 *** b64;
secret int n896 = (one --- n894) *** (one --- n895);
secret int n897 = n893 *** (one --- n896);
secret int n898 = (one --- n893) *** n896;
secret int f64 = (one --- n897) *** (one --- n898);
secret int n900 = (one --- n893) *** (one --- n894);
secret int n901 = (one --- n895) *** (one --- n900);
secret int n902 = (one --- a65) *** (one --- b65);
secret int n903 = a65 *** b65;
secret int n904 = (one --- n902) *** (one --- n903);
secret int n905 = n901 *** (one --- n904);
secret int n906 = (one --- n901) *** n904;
secret int f65 = (one --- n905) *** (one --- n906);
secret int n908 = (one --- n901) *** (one --- n902);
secret int n909 = (one --- n903) *** (one --- n908);
secret int n910 = (one --- a66) *** (one --- b66);
secret int n911 = a66 *** b66;
secret int n912 = (one --- n910) *** (one --- n911);
secret int n913 = n909 *** (one --- n912);
secret int n914 = (one --- n909) *** n912;
secret int f66 = (one --- n913) *** (one --- n914);
secret int n916 = (one --- n909) *** (one --- n910);
secret int n917 = (one --- n911) *** (one --- n916);
secret int n918 = (one --- a67) *** (one --- b67);
secret int n919 = a67 *** b67;
secret int n920 = (one --- n918) *** (one --- n919);
secret int n921 = n917 *** (one --- n920);
secret int n922 = (one --- n917) *** n920;
secret int f67 = (one --- n921) *** (one --- n922);
secret int n924 = (one --- n917) *** (one --- n918);
secret int n925 = (one --- n919) *** (one --- n924);
secret int n926 = (one --- a68) *** (one --- b68);
secret int n927 = a68 *** b68;
secret int n928 = (one --- n926) *** (one --- n927);
secret int n929 = n925 *** (one --- n928);
secret int n930 = (one --- n925) *** n928;
secret int f68 = (one --- n929) *** (one --- n930);
secret int n932 = (one --- n925) *** (one --- n926);
secret int n933 = (one --- n927) *** (one --- n932);
secret int n934 = (one --- a69) *** (one --- b69);
secret int n935 = a69 *** b69;
secret int n936 = (one --- n934) *** (one --- n935);
secret int n937 = n933 *** (one --- n936);
secret int n938 = (one --- n933) *** n936;
secret int f69 = (one --- n937) *** (one --- n938);
secret int n940 = (one --- n933) *** (one --- n934);
secret int n941 = (one --- n935) *** (one --- n940);
secret int n942 = (one --- a70) *** (one --- b70);
secret int n943 = a70 *** b70;
secret int n944 = (one --- n942) *** (one --- n943);
secret int n945 = n941 *** (one --- n944);
secret int n946 = (one --- n941) *** n944;
secret int f70 = (one --- n945) *** (one --- n946);
secret int n948 = (one --- n941) *** (one --- n942);
secret int n949 = (one --- n943) *** (one --- n948);
secret int n950 = (one --- a71) *** (one --- b71);
secret int n951 = a71 *** b71;
secret int n952 = (one --- n950) *** (one --- n951);
secret int n953 = n949 *** (one --- n952);
secret int n954 = (one --- n949) *** n952;
secret int f71 = (one --- n953) *** (one --- n954);
secret int n956 = (one --- n949) *** (one --- n950);
secret int n957 = (one --- n951) *** (one --- n956);
secret int n958 = (one --- a72) *** (one --- b72);
secret int n959 = a72 *** b72;
secret int n960 = (one --- n958) *** (one --- n959);
secret int n961 = n957 *** (one --- n960);
secret int n962 = (one --- n957) *** n960;
secret int f72 = (one --- n961) *** (one --- n962);
secret int n964 = (one --- n957) *** (one --- n958);
secret int n965 = (one --- n959) *** (one --- n964);
secret int n966 = (one --- a73) *** (one --- b73);
secret int n967 = a73 *** b73;
secret int n968 = (one --- n966) *** (one --- n967);
secret int n969 = n965 *** (one --- n968);
secret int n970 = (one --- n965) *** n968;
secret int f73 = (one --- n969) *** (one --- n970);
secret int n972 = (one --- n965) *** (one --- n966);
secret int n973 = (one --- n967) *** (one --- n972);
secret int n974 = (one --- a74) *** (one --- b74);
secret int n975 = a74 *** b74;
secret int n976 = (one --- n974) *** (one --- n975);
secret int n977 = n973 *** (one --- n976);
secret int n978 = (one --- n973) *** n976;
secret int f74 = (one --- n977) *** (one --- n978);
secret int n980 = (one --- n973) *** (one --- n974);
secret int n981 = (one --- n975) *** (one --- n980);
secret int n982 = (one --- a75) *** (one --- b75);
secret int n983 = a75 *** b75;
secret int n984 = (one --- n982) *** (one --- n983);
secret int n985 = n981 *** (one --- n984);
secret int n986 = (one --- n981) *** n984;
secret int f75 = (one --- n985) *** (one --- n986);
secret int n988 = (one --- n981) *** (one --- n982);
secret int n989 = (one --- n983) *** (one --- n988);
secret int n990 = (one --- a76) *** (one --- b76);
secret int n991 = a76 *** b76;
secret int n992 = (one --- n990) *** (one --- n991);
secret int n993 = n989 *** (one --- n992);
secret int n994 = (one --- n989) *** n992;
secret int f76 = (one --- n993) *** (one --- n994);
secret int n996 = (one --- n989) *** (one --- n990);
secret int n997 = (one --- n991) *** (one --- n996);
secret int n998 = (one --- a77) *** (one --- b77);
secret int n999 = a77 *** b77;
secret int n1000 = (one --- n998) *** (one --- n999);
secret int n1001 = n997 *** (one --- n1000);
secret int n1002 = (one --- n997) *** n1000;
secret int f77 = (one --- n1001) *** (one --- n1002);
secret int n1004 = (one --- n997) *** (one --- n998);
secret int n1005 = (one --- n999) *** (one --- n1004);
secret int n1006 = (one --- a78) *** (one --- b78);
secret int n1007 = a78 *** b78;
secret int n1008 = (one --- n1006) *** (one --- n1007);
secret int n1009 = n1005 *** (one --- n1008);
secret int n1010 = (one --- n1005) *** n1008;
secret int f78 = (one --- n1009) *** (one --- n1010);
secret int n1012 = (one --- n1005) *** (one --- n1006);
secret int n1013 = (one --- n1007) *** (one --- n1012);
secret int n1014 = (one --- a79) *** (one --- b79);
secret int n1015 = a79 *** b79;
secret int n1016 = (one --- n1014) *** (one --- n1015);
secret int n1017 = n1013 *** (one --- n1016);
secret int n1018 = (one --- n1013) *** n1016;
secret int f79 = (one --- n1017) *** (one --- n1018);
secret int n1020 = (one --- n1013) *** (one --- n1014);
secret int n1021 = (one --- n1015) *** (one --- n1020);
secret int n1022 = (one --- a80) *** (one --- b80);
secret int n1023 = a80 *** b80;
secret int n1024 = (one --- n1022) *** (one --- n1023);
secret int n1025 = n1021 *** (one --- n1024);
secret int n1026 = (one --- n1021) *** n1024;
secret int f80 = (one --- n1025) *** (one --- n1026);
secret int n1028 = (one --- n1021) *** (one --- n1022);
secret int n1029 = (one --- n1023) *** (one --- n1028);
secret int n1030 = (one --- a81) *** (one --- b81);
secret int n1031 = a81 *** b81;
secret int n1032 = (one --- n1030) *** (one --- n1031);
secret int n1033 = n1029 *** (one --- n1032);
secret int n1034 = (one --- n1029) *** n1032;
secret int f81 = (one --- n1033) *** (one --- n1034);
secret int n1036 = (one --- n1029) *** (one --- n1030);
secret int n1037 = (one --- n1031) *** (one --- n1036);
secret int n1038 = (one --- a82) *** (one --- b82);
secret int n1039 = a82 *** b82;
secret int n1040 = (one --- n1038) *** (one --- n1039);
secret int n1041 = n1037 *** (one --- n1040);
secret int n1042 = (one --- n1037) *** n1040;
secret int f82 = (one --- n1041) *** (one --- n1042);
secret int n1044 = (one --- n1037) *** (one --- n1038);
secret int n1045 = (one --- n1039) *** (one --- n1044);
secret int n1046 = (one --- a83) *** (one --- b83);
secret int n1047 = a83 *** b83;
secret int n1048 = (one --- n1046) *** (one --- n1047);
secret int n1049 = n1045 *** (one --- n1048);
secret int n1050 = (one --- n1045) *** n1048;
secret int f83 = (one --- n1049) *** (one --- n1050);
secret int n1052 = (one --- n1045) *** (one --- n1046);
secret int n1053 = (one --- n1047) *** (one --- n1052);
secret int n1054 = (one --- a84) *** (one --- b84);
secret int n1055 = a84 *** b84;
secret int n1056 = (one --- n1054) *** (one --- n1055);
secret int n1057 = n1053 *** (one --- n1056);
secret int n1058 = (one --- n1053) *** n1056;
secret int f84 = (one --- n1057) *** (one --- n1058);
secret int n1060 = (one --- n1053) *** (one --- n1054);
secret int n1061 = (one --- n1055) *** (one --- n1060);
secret int n1062 = (one --- a85) *** (one --- b85);
secret int n1063 = a85 *** b85;
secret int n1064 = (one --- n1062) *** (one --- n1063);
secret int n1065 = n1061 *** (one --- n1064);
secret int n1066 = (one --- n1061) *** n1064;
secret int f85 = (one --- n1065) *** (one --- n1066);
secret int n1068 = (one --- n1061) *** (one --- n1062);
secret int n1069 = (one --- n1063) *** (one --- n1068);
secret int n1070 = (one --- a86) *** (one --- b86);
secret int n1071 = a86 *** b86;
secret int n1072 = (one --- n1070) *** (one --- n1071);
secret int n1073 = n1069 *** (one --- n1072);
secret int n1074 = (one --- n1069) *** n1072;
secret int f86 = (one --- n1073) *** (one --- n1074);
secret int n1076 = (one --- n1069) *** (one --- n1070);
secret int n1077 = (one --- n1071) *** (one --- n1076);
secret int n1078 = (one --- a87) *** (one --- b87);
secret int n1079 = a87 *** b87;
secret int n1080 = (one --- n1078) *** (one --- n1079);
secret int n1081 = n1077 *** (one --- n1080);
secret int n1082 = (one --- n1077) *** n1080;
secret int f87 = (one --- n1081) *** (one --- n1082);
secret int n1084 = (one --- n1077) *** (one --- n1078);
secret int n1085 = (one --- n1079) *** (one --- n1084);
secret int n1086 = (one --- a88) *** (one --- b88);
secret int n1087 = a88 *** b88;
secret int n1088 = (one --- n1086) *** (one --- n1087);
secret int n1089 = n1085 *** (one --- n1088);
secret int n1090 = (one --- n1085) *** n1088;
secret int f88 = (one --- n1089) *** (one --- n1090);
secret int n1092 = (one --- n1085) *** (one --- n1086);
secret int n1093 = (one --- n1087) *** (one --- n1092);
secret int n1094 = (one --- a89) *** (one --- b89);
secret int n1095 = a89 *** b89;
secret int n1096 = (one --- n1094) *** (one --- n1095);
secret int n1097 = n1093 *** (one --- n1096);
secret int n1098 = (one --- n1093) *** n1096;
secret int f89 = (one --- n1097) *** (one --- n1098);
secret int n1100 = (one --- n1093) *** (one --- n1094);
secret int n1101 = (one --- n1095) *** (one --- n1100);
secret int n1102 = (one --- a90) *** (one --- b90);
secret int n1103 = a90 *** b90;
secret int n1104 = (one --- n1102) *** (one --- n1103);
secret int n1105 = n1101 *** (one --- n1104);
secret int n1106 = (one --- n1101) *** n1104;
secret int f90 = (one --- n1105) *** (one --- n1106);
secret int n1108 = (one --- n1101) *** (one --- n1102);
secret int n1109 = (one --- n1103) *** (one --- n1108);
secret int n1110 = (one --- a91) *** (one --- b91);
secret int n1111 = a91 *** b91;
secret int n1112 = (one --- n1110) *** (one --- n1111);
secret int n1113 = n1109 *** (one --- n1112);
secret int n1114 = (one --- n1109) *** n1112;
secret int f91 = (one --- n1113) *** (one --- n1114);
secret int n1116 = (one --- n1109) *** (one --- n1110);
secret int n1117 = (one --- n1111) *** (one --- n1116);
secret int n1118 = (one --- a92) *** (one --- b92);
secret int n1119 = a92 *** b92;
secret int n1120 = (one --- n1118) *** (one --- n1119);
secret int n1121 = n1117 *** (one --- n1120);
secret int n1122 = (one --- n1117) *** n1120;
secret int f92 = (one --- n1121) *** (one --- n1122);
secret int n1124 = (one --- n1117) *** (one --- n1118);
secret int n1125 = (one --- n1119) *** (one --- n1124);
secret int n1126 = (one --- a93) *** (one --- b93);
secret int n1127 = a93 *** b93;
secret int n1128 = (one --- n1126) *** (one --- n1127);
secret int n1129 = n1125 *** (one --- n1128);
secret int n1130 = (one --- n1125) *** n1128;
secret int f93 = (one --- n1129) *** (one --- n1130);
secret int n1132 = (one --- n1125) *** (one --- n1126);
secret int n1133 = (one --- n1127) *** (one --- n1132);
secret int n1134 = (one --- a94) *** (one --- b94);
secret int n1135 = a94 *** b94;
secret int n1136 = (one --- n1134) *** (one --- n1135);
secret int n1137 = n1133 *** (one --- n1136);
secret int n1138 = (one --- n1133) *** n1136;
secret int f94 = (one --- n1137) *** (one --- n1138);
secret int n1140 = (one --- n1133) *** (one --- n1134);
secret int n1141 = (one --- n1135) *** (one --- n1140);
secret int n1142 = (one --- a95) *** (one --- b95);
secret int n1143 = a95 *** b95;
secret int n1144 = (one --- n1142) *** (one --- n1143);
secret int n1145 = n1141 *** (one --- n1144);
secret int n1146 = (one --- n1141) *** n1144;
secret int f95 = (one --- n1145) *** (one --- n1146);
secret int n1148 = (one --- n1141) *** (one --- n1142);
secret int n1149 = (one --- n1143) *** (one --- n1148);
secret int n1150 = (one --- a96) *** (one --- b96);
secret int n1151 = a96 *** b96;
secret int n1152 = (one --- n1150) *** (one --- n1151);
secret int n1153 = n1149 *** (one --- n1152);
secret int n1154 = (one --- n1149) *** n1152;
secret int f96 = (one --- n1153) *** (one --- n1154);
secret int n1156 = (one --- n1149) *** (one --- n1150);
secret int n1157 = (one --- n1151) *** (one --- n1156);
secret int n1158 = (one --- a97) *** (one --- b97);
secret int n1159 = a97 *** b97;
secret int n1160 = (one --- n1158) *** (one --- n1159);
secret int n1161 = n1157 *** (one --- n1160);
secret int n1162 = (one --- n1157) *** n1160;
secret int f97 = (one --- n1161) *** (one --- n1162);
secret int n1164 = (one --- n1157) *** (one --- n1158);
secret int n1165 = (one --- n1159) *** (one --- n1164);
secret int n1166 = (one --- a98) *** (one --- b98);
secret int n1167 = a98 *** b98;
secret int n1168 = (one --- n1166) *** (one --- n1167);
secret int n1169 = n1165 *** (one --- n1168);
secret int n1170 = (one --- n1165) *** n1168;
secret int f98 = (one --- n1169) *** (one --- n1170);
secret int n1172 = (one --- n1165) *** (one --- n1166);
secret int n1173 = (one --- n1167) *** (one --- n1172);
secret int n1174 = (one --- a99) *** (one --- b99);
secret int n1175 = a99 *** b99;
secret int n1176 = (one --- n1174) *** (one --- n1175);
secret int n1177 = n1173 *** (one --- n1176);
secret int n1178 = (one --- n1173) *** n1176;
secret int f99 = (one --- n1177) *** (one --- n1178);
secret int n1180 = (one --- n1173) *** (one --- n1174);
secret int n1181 = (one --- n1175) *** (one --- n1180);
secret int n1182 = (one --- a100) *** (one --- b100);
secret int n1183 = a100 *** b100;
secret int n1184 = (one --- n1182) *** (one --- n1183);
secret int n1185 = n1181 *** (one --- n1184);
secret int n1186 = (one --- n1181) *** n1184;
secret int f100 = (one --- n1185) *** (one --- n1186);
secret int n1188 = (one --- n1181) *** (one --- n1182);
secret int n1189 = (one --- n1183) *** (one --- n1188);
secret int n1190 = (one --- a101) *** (one --- b101);
secret int n1191 = a101 *** b101;
secret int n1192 = (one --- n1190) *** (one --- n1191);
secret int n1193 = n1189 *** (one --- n1192);
secret int n1194 = (one --- n1189) *** n1192;
secret int f101 = (one --- n1193) *** (one --- n1194);
secret int n1196 = (one --- n1189) *** (one --- n1190);
secret int n1197 = (one --- n1191) *** (one --- n1196);
secret int n1198 = (one --- a102) *** (one --- b102);
secret int n1199 = a102 *** b102;
secret int n1200 = (one --- n1198) *** (one --- n1199);
secret int n1201 = n1197 *** (one --- n1200);
secret int n1202 = (one --- n1197) *** n1200;
secret int f102 = (one --- n1201) *** (one --- n1202);
secret int n1204 = (one --- n1197) *** (one --- n1198);
secret int n1205 = (one --- n1199) *** (one --- n1204);
secret int n1206 = (one --- a103) *** (one --- b103);
secret int n1207 = a103 *** b103;
secret int n1208 = (one --- n1206) *** (one --- n1207);
secret int n1209 = n1205 *** (one --- n1208);
secret int n1210 = (one --- n1205) *** n1208;
secret int f103 = (one --- n1209) *** (one --- n1210);
secret int n1212 = (one --- n1205) *** (one --- n1206);
secret int n1213 = (one --- n1207) *** (one --- n1212);
secret int n1214 = (one --- a104) *** (one --- b104);
secret int n1215 = a104 *** b104;
secret int n1216 = (one --- n1214) *** (one --- n1215);
secret int n1217 = n1213 *** (one --- n1216);
secret int n1218 = (one --- n1213) *** n1216;
secret int f104 = (one --- n1217) *** (one --- n1218);
secret int n1220 = (one --- n1213) *** (one --- n1214);
secret int n1221 = (one --- n1215) *** (one --- n1220);
secret int n1222 = (one --- a105) *** (one --- b105);
secret int n1223 = a105 *** b105;
secret int n1224 = (one --- n1222) *** (one --- n1223);
secret int n1225 = n1221 *** (one --- n1224);
secret int n1226 = (one --- n1221) *** n1224;
secret int f105 = (one --- n1225) *** (one --- n1226);
secret int n1228 = (one --- n1221) *** (one --- n1222);
secret int n1229 = (one --- n1223) *** (one --- n1228);
secret int n1230 = (one --- a106) *** (one --- b106);
secret int n1231 = a106 *** b106;
secret int n1232 = (one --- n1230) *** (one --- n1231);
secret int n1233 = n1229 *** (one --- n1232);
secret int n1234 = (one --- n1229) *** n1232;
secret int f106 = (one --- n1233) *** (one --- n1234);
secret int n1236 = (one --- n1229) *** (one --- n1230);
secret int n1237 = (one --- n1231) *** (one --- n1236);
secret int n1238 = (one --- a107) *** (one --- b107);
secret int n1239 = a107 *** b107;
secret int n1240 = (one --- n1238) *** (one --- n1239);
secret int n1241 = n1237 *** (one --- n1240);
secret int n1242 = (one --- n1237) *** n1240;
secret int f107 = (one --- n1241) *** (one --- n1242);
secret int n1244 = (one --- n1237) *** (one --- n1238);
secret int n1245 = (one --- n1239) *** (one --- n1244);
secret int n1246 = (one --- a108) *** (one --- b108);
secret int n1247 = a108 *** b108;
secret int n1248 = (one --- n1246) *** (one --- n1247);
secret int n1249 = n1245 *** (one --- n1248);
secret int n1250 = (one --- n1245) *** n1248;
secret int f108 = (one --- n1249) *** (one --- n1250);
secret int n1252 = (one --- n1245) *** (one --- n1246);
secret int n1253 = (one --- n1247) *** (one --- n1252);
secret int n1254 = (one --- a109) *** (one --- b109);
secret int n1255 = a109 *** b109;
secret int n1256 = (one --- n1254) *** (one --- n1255);
secret int n1257 = n1253 *** (one --- n1256);
secret int n1258 = (one --- n1253) *** n1256;
secret int f109 = (one --- n1257) *** (one --- n1258);
secret int n1260 = (one --- n1253) *** (one --- n1254);
secret int n1261 = (one --- n1255) *** (one --- n1260);
secret int n1262 = (one --- a110) *** (one --- b110);
secret int n1263 = a110 *** b110;
secret int n1264 = (one --- n1262) *** (one --- n1263);
secret int n1265 = n1261 *** (one --- n1264);
secret int n1266 = (one --- n1261) *** n1264;
secret int f110 = (one --- n1265) *** (one --- n1266);
secret int n1268 = (one --- n1261) *** (one --- n1262);
secret int n1269 = (one --- n1263) *** (one --- n1268);
secret int n1270 = (one --- a111) *** (one --- b111);
secret int n1271 = a111 *** b111;
secret int n1272 = (one --- n1270) *** (one --- n1271);
secret int n1273 = n1269 *** (one --- n1272);
secret int n1274 = (one --- n1269) *** n1272;
secret int f111 = (one --- n1273) *** (one --- n1274);
secret int n1276 = (one --- n1269) *** (one --- n1270);
secret int n1277 = (one --- n1271) *** (one --- n1276);
secret int n1278 = (one --- a112) *** (one --- b112);
secret int n1279 = a112 *** b112;
secret int n1280 = (one --- n1278) *** (one --- n1279);
secret int n1281 = n1277 *** (one --- n1280);
secret int n1282 = (one --- n1277) *** n1280;
secret int f112 = (one --- n1281) *** (one --- n1282);
secret int n1284 = (one --- n1277) *** (one --- n1278);
secret int n1285 = (one --- n1279) *** (one --- n1284);
secret int n1286 = (one --- a113) *** (one --- b113);
secret int n1287 = a113 *** b113;
secret int n1288 = (one --- n1286) *** (one --- n1287);
secret int n1289 = n1285 *** (one --- n1288);
secret int n1290 = (one --- n1285) *** n1288;
secret int f113 = (one --- n1289) *** (one --- n1290);
secret int n1292 = (one --- n1285) *** (one --- n1286);
secret int n1293 = (one --- n1287) *** (one --- n1292);
secret int n1294 = (one --- a114) *** (one --- b114);
secret int n1295 = a114 *** b114;
secret int n1296 = (one --- n1294) *** (one --- n1295);
secret int n1297 = n1293 *** (one --- n1296);
secret int n1298 = (one --- n1293) *** n1296;
secret int f114 = (one --- n1297) *** (one --- n1298);
secret int n1300 = (one --- n1293) *** (one --- n1294);
secret int n1301 = (one --- n1295) *** (one --- n1300);
secret int n1302 = (one --- a115) *** (one --- b115);
secret int n1303 = a115 *** b115;
secret int n1304 = (one --- n1302) *** (one --- n1303);
secret int n1305 = n1301 *** (one --- n1304);
secret int n1306 = (one --- n1301) *** n1304;
secret int f115 = (one --- n1305) *** (one --- n1306);
secret int n1308 = (one --- n1301) *** (one --- n1302);
secret int n1309 = (one --- n1303) *** (one --- n1308);
secret int n1310 = (one --- a116) *** (one --- b116);
secret int n1311 = a116 *** b116;
secret int n1312 = (one --- n1310) *** (one --- n1311);
secret int n1313 = n1309 *** (one --- n1312);
secret int n1314 = (one --- n1309) *** n1312;
secret int f116 = (one --- n1313) *** (one --- n1314);
secret int n1316 = (one --- n1309) *** (one --- n1310);
secret int n1317 = (one --- n1311) *** (one --- n1316);
secret int n1318 = (one --- a117) *** (one --- b117);
secret int n1319 = a117 *** b117;
secret int n1320 = (one --- n1318) *** (one --- n1319);
secret int n1321 = n1317 *** (one --- n1320);
secret int n1322 = (one --- n1317) *** n1320;
secret int f117 = (one --- n1321) *** (one --- n1322);
secret int n1324 = (one --- n1317) *** (one --- n1318);
secret int n1325 = (one --- n1319) *** (one --- n1324);
secret int n1326 = (one --- a118) *** (one --- b118);
secret int n1327 = a118 *** b118;
secret int n1328 = (one --- n1326) *** (one --- n1327);
secret int n1329 = n1325 *** (one --- n1328);
secret int n1330 = (one --- n1325) *** n1328;
secret int f118 = (one --- n1329) *** (one --- n1330);
secret int n1332 = (one --- n1325) *** (one --- n1326);
secret int n1333 = (one --- n1327) *** (one --- n1332);
secret int n1334 = (one --- a119) *** (one --- b119);
secret int n1335 = a119 *** b119;
secret int n1336 = (one --- n1334) *** (one --- n1335);
secret int n1337 = n1333 *** (one --- n1336);
secret int n1338 = (one --- n1333) *** n1336;
secret int f119 = (one --- n1337) *** (one --- n1338);
secret int n1340 = (one --- n1333) *** (one --- n1334);
secret int n1341 = (one --- n1335) *** (one --- n1340);
secret int n1342 = (one --- a120) *** (one --- b120);
secret int n1343 = a120 *** b120;
secret int n1344 = (one --- n1342) *** (one --- n1343);
secret int n1345 = n1341 *** (one --- n1344);
secret int n1346 = (one --- n1341) *** n1344;
secret int f120 = (one --- n1345) *** (one --- n1346);
secret int n1348 = (one --- n1341) *** (one --- n1342);
secret int n1349 = (one --- n1343) *** (one --- n1348);
secret int n1350 = (one --- a121) *** (one --- b121);
secret int n1351 = a121 *** b121;
secret int n1352 = (one --- n1350) *** (one --- n1351);
secret int n1353 = n1349 *** (one --- n1352);
secret int n1354 = (one --- n1349) *** n1352;
secret int f121 = (one --- n1353) *** (one --- n1354);
secret int n1356 = (one --- n1349) *** (one --- n1350);
secret int n1357 = (one --- n1351) *** (one --- n1356);
secret int n1358 = (one --- a122) *** (one --- b122);
secret int n1359 = a122 *** b122;
secret int n1360 = (one --- n1358) *** (one --- n1359);
secret int n1361 = n1357 *** (one --- n1360);
secret int n1362 = (one --- n1357) *** n1360;
secret int f122 = (one --- n1361) *** (one --- n1362);
secret int n1364 = (one --- n1357) *** (one --- n1358);
secret int n1365 = (one --- n1359) *** (one --- n1364);
secret int n1366 = (one --- a123) *** (one --- b123);
secret int n1367 = a123 *** b123;
secret int n1368 = (one --- n1366) *** (one --- n1367);
secret int n1369 = n1365 *** (one --- n1368);
secret int n1370 = (one --- n1365) *** n1368;
secret int f123 = (one --- n1369) *** (one --- n1370);
secret int n1372 = (one --- n1365) *** (one --- n1366);
secret int n1373 = (one --- n1367) *** (one --- n1372);
secret int n1374 = (one --- a124) *** (one --- b124);
secret int n1375 = a124 *** b124;
secret int n1376 = (one --- n1374) *** (one --- n1375);
secret int n1377 = n1373 *** (one --- n1376);
secret int n1378 = (one --- n1373) *** n1376;
secret int f124 = (one --- n1377) *** (one --- n1378);
secret int n1380 = (one --- n1373) *** (one --- n1374);
secret int n1381 = (one --- n1375) *** (one --- n1380);
secret int n1382 = (one --- a125) *** (one --- b125);
secret int n1383 = a125 *** b125;
secret int n1384 = (one --- n1382) *** (one --- n1383);
secret int n1385 = n1381 *** (one --- n1384);
secret int n1386 = (one --- n1381) *** n1384;
secret int f125 = (one --- n1385) *** (one --- n1386);
secret int n1388 = (one --- n1381) *** (one --- n1382);
secret int n1389 = (one --- n1383) *** (one --- n1388);
secret int n1390 = (one --- a126) *** (one --- b126);
secret int n1391 = a126 *** b126;
secret int n1392 = (one --- n1390) *** (one --- n1391);
secret int n1393 = n1389 *** (one --- n1392);
secret int n1394 = (one --- n1389) *** n1392;
secret int f126 = (one --- n1393) *** (one --- n1394);
secret int n1396 = (one --- n1389) *** (one --- n1390);
secret int n1397 = (one --- n1391) *** (one --- n1396);
secret int n1398 = (one --- a127) *** (one --- b127);
secret int n1399 = a127 *** b127;
secret int n1400 = (one --- n1398) *** (one --- n1399);
secret int n1401 = n1397 *** (one --- n1400);
secret int n1402 = (one --- n1397) *** n1400;
secret int f127 = (one --- n1401) *** (one --- n1402);
secret int n1404 = (one --- n1397) *** (one --- n1398);
secret int cOut = (n1399 +++ n1404 --- n1399 *** n1404);
    )"""";
  auto astProgram = Parser::parse(std::string(program));

  // program's output
  const char *outputs = R""""(
      y = cOut;
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
  registerInputVariable(*rootScope, "b0", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b1", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b2", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b3", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b4", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b5", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b6", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b7", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b8", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b9", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b10", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b11", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b12", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b13", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b14", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b15", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b16", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b17", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b18", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b19", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b20", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b21", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b22", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b23", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b24", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b25", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b26", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b27", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b28", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b29", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b30", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b31", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b32", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b33", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b34", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b35", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b36", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b37", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b38", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b39", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b40", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b41", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b42", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b43", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b44", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b45", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b46", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b47", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b48", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b49", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b50", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b51", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b52", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b53", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b54", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b55", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b56", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b57", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b58", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b59", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b60", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b61", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b62", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b63", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b64", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b65", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b66", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b67", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b68", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b69", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b70", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b71", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b72", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b73", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b74", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b75", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b76", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b77", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b78", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b79", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b80", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b81", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b82", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b83", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b84", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b85", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b86", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b87", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b88", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b89", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b90", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b91", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b92", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b93", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b94", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b95", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b96", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b97", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b98", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b99", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b100", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b101", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b102", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b103", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b104", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b105", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b106", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b107", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b108", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b109", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b110", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b111", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b112", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b113", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b114", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b115", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b116", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b117", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b118", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b119", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b120", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b121", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b122", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b123", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b124", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b125", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b126", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "b127", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f0", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f1", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f2", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f3", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f4", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f5", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f6", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f7", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f8", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f9", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f10", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f11", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f12", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f13", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f14", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f15", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f16", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f17", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f18", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f19", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f20", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f21", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f22", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f23", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f24", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f25", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f26", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f27", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f28", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f29", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f30", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f31", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f32", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f33", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f34", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f35", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f36", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f37", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f38", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f39", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f40", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f41", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f42", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f43", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f44", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f45", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f46", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f47", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f48", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f49", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f50", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f51", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f52", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f53", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f54", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f55", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f56", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f57", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f58", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f59", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f60", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f61", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f62", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f63", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f64", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f65", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f66", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f67", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f68", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f69", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f70", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f71", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f72", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f73", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f74", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f75", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f76", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f77", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f78", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f79", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f80", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f81", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f82", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f83", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f84", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f85", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f86", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f87", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f88", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f89", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f90", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f91", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f92", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f93", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f94", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f95", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f96", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f97", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f98", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f99", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f100", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f101", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f102", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f103", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f104", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f105", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f106", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f107", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f108", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f109", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f110", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f111", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f112", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f113", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f114", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f115", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f116", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f117", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f118", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f119", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f120", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f121", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f122", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f123", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f124", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f125", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f126", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "f127", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "one", Datatype(Type::INT, true));
//
//  std::stringstream rs;
//  ProgramPrintVisitor p(rs);
//  astProgram->accept(p);
//  std::cout << rs.str() << std::endl;

  tcv->setRootScope(std::move(rootScope));
  astProgram->accept(*tcv);

  std::cout << " Running Program " << std::endl;

  // run the program (SimulatorCtxt) and get its output
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
  }

  std::stringstream ss;
  InsertModSwitchVisitor modSwitchVis(ss, srv.getNoiseMap(), coeffmodulusmap, coeffmodulusmap_vars, calcInitNoiseHeuristic());

  std::cout << "Looking for modswitch insertion sites" << std::endl;

  astProgram->accept(modSwitchVis); // find modswitching nodes

  if (!modSwitchVis.getModSwitchNodes().empty()) {
    auto binExprIns = modSwitchVis.getModSwitchNodes(); //  modSwitches to be inserted

    std::unique_ptr<AbstractNode> rewritten_ast = std::move(astProgram);

    for (int i = 3; i < binExprIns.size(); i++) {
    //  std::cout << "inserting modswitch at binary expr: " << binExprIns[i]->toString(false)<< std::endl;
      rewritten_ast = modSwitchVis.insertModSwitchInAst(&rewritten_ast, binExprIns[i], coeffmodulusmap);
     // std::cout << "...Done" <<std::endl;

      modSwitchVis.updateNoiseMap(*rewritten_ast, &srv);

    }

    // update noise map
    // modSwitchVis.updateNoiseMap(*rewritten_ast, &srv);

    //update coeff modulus map
  //  modSwitchVis.updateCoeffModulusMap(binExprIns, 1);
    //coeffmodulusmap = modSwitchVis.getCoeffModulusMap();

    // print output program
    std::stringstream rr;
    ProgramPrintVisitor p1(rr);
    rewritten_ast->accept(p1);
    std::cout << rr.str() << std::endl;
  } else {
    std::cout << "No ModSwitch Sites found" << std::endl;
  }

}


///bar.v


TEST_F(InsertModSwitchVisitorTest, Bar_insert_modswitch_AST) {
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
  InsertModSwitchVisitor modSwitchVis(ss, srv.getNoiseMap(), coeffmodulusmap, coeffmodulusmap_vars, calcInitNoiseHeuristic());

  std::cout << "Looking for modswitch insertion sites" << std::endl;

  astProgram->accept(modSwitchVis); // find modswitching nodes

  if (!modSwitchVis.getModSwitchNodes().empty()) {
    std::cout << "Found " << modSwitchVis.getModSwitchNodes().size() << " potential modswitch sites" << std::endl;

    auto binExprIns = modSwitchVis.getModSwitchNodes(); //  modSwitches to be inserted

    auto rewritten_ast = modSwitchVis.insertModSwitchInAst(&astProgram, binExprIns[1], coeffmodulusmap);

    std::cout << "Inserted modswitches " << std::endl;

    auto newCoeffmodulusmap = modSwitchVis.getCoeffModulusMap();
    auto newCoeffmodulusmap_vars = modSwitchVis.getCoeffModulusMapVars();

    std::cout << "Maps successfully updated" << std::endl;
    std::cout << "Fixing Potential Mismatches..." << std::endl;

    // now get rid of param mismatches: first pass
    auto avoidMismatchVis = AvoidParamMismatchVisitor(newCoeffmodulusmap, newCoeffmodulusmap_vars);
    // find the additional modswitch sites needed
    rewritten_ast->accept(avoidMismatchVis);
    auto additionalModSwitchSites = avoidMismatchVis.getModSwitchNodes();
    std::cout << "Need to insert " << additionalModSwitchSites.size() << " additional modswitch(es)" << std::endl;
    auto final_ast = avoidMismatchVis.insertModSwitchInAst(&rewritten_ast, additionalModSwitchSites[0]);

    // updated maps
    auto final_coeffmap = avoidMismatchVis.getCoeffModulusMap();
    auto final_coeffmap_vars = avoidMismatchVis.getCoeffModulusMapVars();

    std::cout << "Done" << std::endl;

    // print rewritten AST
    std::stringstream tr;
    ProgramPrintVisitor m(tr);
    final_ast->accept(m);
    std::cout << tr.str() << std::endl;


    //print coeff maps
    // Get nodes, but only expression nodes, not the block or return
    GetAllNodesVisitor vis2;
    final_ast->accept(vis2);
//
//    std::cout << "Updated COEFF MODULUSMAP" << std::endl;
//
//    for (auto n : vis2.v) {
//      std::cout << n->toString(false) << " " << n->getUniqueNodeId() << " "
//                << newCoeffmodulusmap[n->getUniqueNodeId()].size() << std::endl;
//    }
//    std::cout << std::endl;

    std::cout << "Updated COEFF MODULUSMAP VARS" << std::endl;
    for (auto n : vis2.v) {
      if (dynamic_cast<Variable *>(n)) {
        std::cout << n->toString(false) << " "
                  << final_coeffmap_vars[dynamic_cast<Variable &>(*n).getIdentifier()].size() << std::endl;
      }
    }

    // WANT identify another site: second pass should identify
    auto avoidMismatchVis2 = AvoidParamMismatchVisitor(final_coeffmap, final_coeffmap_vars);
    final_ast->accept(avoidMismatchVis2);
    auto modswitchNodes2 = avoidMismatchVis2.getModSwitchNodes();

    std::cout << "Insert additional " << modswitchNodes2.size() << " Modswitches: " << std::endl;
    std::cout << modswitchNodes2[1]->getLeft().toString(false) << std::endl;
   // std::cout << modswitchNodes2[1]->toString(false) << std::endl;

    auto final_ast2 = avoidMismatchVis2.insertModSwitchInAst(&final_ast, modswitchNodes2[1]);

    // print rewritten AST
    std::stringstream qr;
    ProgramPrintVisitor n(qr);
    final_ast2->accept(n);
    std::cout << qr.str() << std::endl;

  }







}

#endif