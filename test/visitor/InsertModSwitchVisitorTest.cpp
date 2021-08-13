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
    size_t coeff_modulus_significant_bit_count = scf->getContext().first_context_data()->total_coeff_modulus_bit_count();
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
  for (auto n : vis.v) {
    coeffmodulusmap[n->getUniqueNodeId()] = coeff_modulus;
  }

  std::stringstream rr;
  InsertModSwitchVisitor modSwitchVis(rr, srv.getNoiseMap(), coeffmodulusmap, calcInitNoiseHeuristic());

  astProgram->accept(modSwitchVis); // find modswitching nodes

  std::cout << modSwitchVis.getModSwitchNodes().size() << " potential modSwitch insertion site(s) found:" << std::endl;
  for (int j = 0; j < modSwitchVis.getModSwitchNodes().size(); j++) {
    std::cout << modSwitchVis.getModSwitchNodes()[j]->toString(false) << " " <<  modSwitchVis.getModSwitchNodes()[j]->getUniqueNodeId() << std::endl;
  }

  EXPECT_EQ( modSwitchVis.getModSwitchNodes()[0]->getUniqueNodeId(), "BinaryExpression_106");

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
  for (auto n : vis.v) {
    coeffmodulusmap[n->getUniqueNodeId()] = coeff_modulus;
  }

  std::stringstream rr;
  InsertModSwitchVisitor modSwitchVis(rr, srv.getNoiseMap(), coeffmodulusmap, calcInitNoiseHeuristic());

  astProgram->accept(modSwitchVis); // find modswitching nodes

  std::cout << modSwitchVis.getModSwitchNodes().size() << " potential modSwitch insertion site(s) found:" << std::endl;
  for (int j = 0; j < modSwitchVis.getModSwitchNodes().size(); j++) {
    std::cout << modSwitchVis.getModSwitchNodes()[j]->toString(false) << " " <<  modSwitchVis.getModSwitchNodes()[j]->getUniqueNodeId() << std::endl;
  }

  EXPECT_EQ(modSwitchVis.getModSwitchNodes().size(), 0);

}

TEST_F(InsertModSwitchVisitorTest, rewriteASTnoChangeExpected) {
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
  for (auto n : vis.v) {
    coeffmodulusmap[n->getUniqueNodeId()] = coeff_modulus;
  }

  std::stringstream ss;
  InsertModSwitchVisitor modSwitchVis(ss, srv.getNoiseMap(), coeffmodulusmap, calcInitNoiseHeuristic());

  astProgram->accept(modSwitchVis); // find modswitching nodes


  BinaryExpression *binExprIns = nullptr;

  if (!modSwitchVis.getModSwitchNodes().empty()) {
    binExprIns = modSwitchVis.getModSwitchNodes()[0]; //  modSwitches to be inserted
    std::cout << "TEST: " <<  binExprIns->getUniqueNodeId() << std::endl;
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

TEST_F(InsertModSwitchVisitorTest, rewriteASTmodSwitchBeforeLastBinaryOpExpected) {

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
  for (auto n : vis.v) {
    coeffmodulusmap[n->getUniqueNodeId()] = coeff_modulus;
  }

  std::stringstream ss;
  InsertModSwitchVisitor modSwitchVis(ss, srv.getNoiseMap(), coeffmodulusmap, calcInitNoiseHeuristic());

  astProgram->accept(modSwitchVis); // find modswitching nodes

  auto binExprIns = modSwitchVis.getModSwitchNodes()[0]; //  modSwitches to be inserted

  auto rewritten_ast = modSwitchVis.insertModSwitchInAst(&astProgram, binExprIns, coeffmodulusmap);

  // update noise map
  modSwitchVis.updateNoiseMap(*rewritten_ast, &srv);

  //update coeff modulus map
  modSwitchVis.updateCoeffModulusMap(binExprIns,1);
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

TEST_F(InsertModSwitchVisitorTest, rewriteASTTwomodSwitchesBeforeLastBinaryOpExpected) {

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
  InsertModSwitchVisitor modSwitchVis(rr, tamperedNoiseMap, coeffmodulusmap, calcInitNoiseHeuristic());
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

  std::cout << "Here:" << std::endl;

//  std::stringstream ss;
//  PrintVisitor p(ss);
//  astProgram->accept(p);
//  std::cout << ss.str() << std::endl;

  // WANT: remove BinaryExpression_40
  // TODO: remove modswitches


  //expected circuit
  // program specification
  const char *expected_program = R""""(
      secret int result = (__input0__ ***  __input1__);
      return result;
    )"""";
  auto expected_astProgram = Parser::parse(std::string(expected_program));

  //compareAST(*rewritten_ast, *expected_astProgram);

 EXPECT_EQ(true, false);

}


TEST_F(InsertModSwitchVisitorTest, AdderAST) {

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
    )"""";
  auto astInput = Parser::parse(std::string(inputs));

  // program specification
  const char *program = R""""(
secret int n386 = a0 *** (1 --- b0);
secret int n387 = (1 --- a0) *** b0;
secret int f0 = (n386 +++ n387 --- n386 *** n387);
secret int n389 = a0 *** b0;
secret int n390 = (1 --- a1) *** (1 --- b1);
secret int n391 = a1 *** b1;
secret int n392 = (1 --- n390) *** (1 --- n391);
secret int n393 = n389 *** (1 --- n392);
secret int n394 = (1 --- n389) *** n392;
secret int f1 = (n393 +++ n394 --- n393 *** n394);
secret int n396 = n389 *** (1 --- n390);
secret int n397 = (1 --- n391) *** (1 --- n396);
secret int n398 = (1 --- a2) *** (1 --- b2);
secret int n399 = a2 *** b2;
secret int n400 = (1 --- n398) *** (1 --- n399);
secret int n401 = n397 *** (1 --- n400);
secret int n402 = (1 --- n397) *** n400;
secret int f2 = (1 --- n401) *** (1 --- n402);
secret int n404 = (1 --- n397) *** (1 --- n398);
secret int n405 = (1 --- n399) *** (1 --- n404);
secret int n406 = (1 --- a3) *** (1 --- b3);
secret int n407 = a3 *** b3;
secret int n408 = (1 --- n406) *** (1 --- n407);
secret int n409 = n405 *** (1 --- n408);
secret int n410 = (1 --- n405) *** n408;
secret int f3 = (1 --- n409) *** (1 --- n410);
secret int n412 = (1 --- n405) *** (1 --- n406);
secret int n413 = (1 --- n407) *** (1 --- n412);
secret int n414 = (1 --- a4) *** (1 --- b4);
secret int n415 = a4 *** b4;
secret int n416 = (1 --- n414) *** (1 --- n415);
secret int n417 = n413 *** (1 --- n416);
secret int n418 = (1 --- n413) *** n416;
secret int f4 = (1 --- n417) *** (1 --- n418);
secret int n420 = (1 --- n413) *** (1 --- n414);
secret int n421 = (1 --- n415) *** (1 --- n420);
secret int n422 = (1 --- a5) *** (1 --- b5);
secret int n423 = a5 *** b5;
secret int n424 = (1 --- n422) *** (1 --- n423);
secret int n425 = n421 *** (1 --- n424);
secret int n426 = (1 --- n421) *** n424;
secret int f5 = (1 --- n425) *** (1 --- n426);
secret int n428 = (1 --- n421) *** (1 --- n422);
secret int n429 = (1 --- n423) *** (1 --- n428);
secret int n430 = (1 --- a6) *** (1 --- b6);
secret int n431 = a6 *** b6;
secret int n432 = (1 --- n430) *** (1 --- n431);
secret int n433 = n429 *** (1 --- n432);
secret int n434 = (1 --- n429) *** n432;
secret int f6 = (1 --- n433) *** (1 --- n434);
secret int n436 = (1 --- n429) *** (1 --- n430);
secret int n437 = (1 --- n431) *** (1 --- n436);
secret int n438 = (1 --- a7) *** (1 --- b7);
secret int n439 = a7 *** b7;
secret int n440 = (1 --- n438) *** (1 --- n439);
secret int n441 = n437 *** (1 --- n440);
secret int n442 = (1 --- n437) *** n440;
secret int f7 = (1 --- n441) *** (1 --- n442);
secret int n444 = (1 --- n437) *** (1 --- n438);
secret int n445 = (1 --- n439) *** (1 --- n444);
secret int n446 = (1 --- a8) *** (1 --- b8);
secret int n447 = a8 *** b8;
secret int n448 = (1 --- n446) *** (1 --- n447);
secret int n449 = n445 *** (1 --- n448);
secret int n450 = (1 --- n445) *** n448;
secret int f8 = (1 --- n449) *** (1 --- n450);
secret int n452 = (1 --- n445) *** (1 --- n446);
secret int n453 = (1 --- n447) *** (1 --- n452);
secret int n454 = (1 --- a9) *** (1 --- b9);
secret int n455 = a9 *** b9;
secret int n456 = (1 --- n454) *** (1 --- n455);
secret int n457 = n453 *** (1 --- n456);
secret int n458 = (1 --- n453) *** n456;
secret int f9 = (1 --- n457) *** (1 --- n458);
secret int n460 = (1 --- n453) *** (1 --- n454);
secret int n461 = (1 --- n455) *** (1 --- n460);
secret int n462 = (1 --- a10) *** (1 --- b10);
secret int n463 = a10 *** b10;
secret int n464 = (1 --- n462) *** (1 --- n463);
secret int n465 = n461 *** (1 --- n464);
secret int n466 = (1 --- n461) *** n464;
secret int f10 = (1 --- n465) *** (1 --- n466);
secret int n468 = (1 --- n461) *** (1 --- n462);
secret int n469 = (1 --- n463) *** (1 --- n468);
secret int n470 = (1 --- a11) *** (1 --- b11);
secret int n471 = a11 *** b11;
secret int n472 = (1 --- n470) *** (1 --- n471);
secret int n473 = n469 *** (1 --- n472);
secret int n474 = (1 --- n469) *** n472;
secret int f11 = (1 --- n473) *** (1 --- n474);
secret int n476 = (1 --- n469) *** (1 --- n470);
secret int n477 = (1 --- n471) *** (1 --- n476);
secret int n478 = (1 --- a12) *** (1 --- b12);
secret int n479 = a12 *** b12;
secret int n480 = (1 --- n478) *** (1 --- n479);
secret int n481 = n477 *** (1 --- n480);
secret int n482 = (1 --- n477) *** n480;
secret int f12 = (1 --- n481) *** (1 --- n482);
secret int n484 = (1 --- n477) *** (1 --- n478);
secret int n485 = (1 --- n479) *** (1 --- n484);
secret int n486 = (1 --- a13) *** (1 --- b13);
secret int n487 = a13 *** b13;
secret int n488 = (1 --- n486) *** (1 --- n487);
secret int n489 = n485 *** (1 --- n488);
secret int n490 = (1 --- n485) *** n488;
secret int f13 = (1 --- n489) *** (1 --- n490);
secret int n492 = (1 --- n485) *** (1 --- n486);
secret int n493 = (1 --- n487) *** (1 --- n492);
secret int n494 = (1 --- a14) *** (1 --- b14);
secret int n495 = a14 *** b14;
secret int n496 = (1 --- n494) *** (1 --- n495);
secret int n497 = n493 *** (1 --- n496);
secret int n498 = (1 --- n493) *** n496;
secret int f14 = (1 --- n497) *** (1 --- n498);
secret int n500 = (1 --- n493) *** (1 --- n494);
secret int n501 = (1 --- n495) *** (1 --- n500);
secret int n502 = (1 --- a15) *** (1 --- b15);
secret int n503 = a15 *** b15;
secret int n504 = (1 --- n502) *** (1 --- n503);
secret int n505 = n501 *** (1 --- n504);
secret int n506 = (1 --- n501) *** n504;
secret int f15 = (1 --- n505) *** (1 --- n506);
secret int n508 = (1 --- n501) *** (1 --- n502);
secret int n509 = (1 --- n503) *** (1 --- n508);
secret int n510 = (1 --- a16) *** (1 --- b16);
secret int n511 = a16 *** b16;
secret int n512 = (1 --- n510) *** (1 --- n511);
secret int n513 = n509 *** (1 --- n512);
secret int n514 = (1 --- n509) *** n512;
secret int f16 = (1 --- n513) *** (1 --- n514);
secret int n516 = (1 --- n509) *** (1 --- n510);
secret int n517 = (1 --- n511) *** (1 --- n516);
secret int n518 = (1 --- a17) *** (1 --- b17);
secret int n519 = a17 *** b17;
secret int n520 = (1 --- n518) *** (1 --- n519);
secret int n521 = n517 *** (1 --- n520);
secret int n522 = (1 --- n517) *** n520;
secret int f17 = (1 --- n521) *** (1 --- n522);
secret int n524 = (1 --- n517) *** (1 --- n518);
secret int n525 = (1 --- n519) *** (1 --- n524);
secret int n526 = (1 --- a18) *** (1 --- b18);
secret int n527 = a18 *** b18;
secret int n528 = (1 --- n526) *** (1 --- n527);
secret int n529 = n525 *** (1 --- n528);
secret int n530 = (1 --- n525) *** n528;
secret int f18 = (1 --- n529) *** (1 --- n530);
secret int n532 = (1 --- n525) *** (1 --- n526);
secret int n533 = (1 --- n527) *** (1 --- n532);
secret int n534 = (1 --- a19) *** (1 --- b19);
secret int n535 = a19 *** b19;
secret int n536 = (1 --- n534) *** (1 --- n535);
secret int n537 = n533 *** (1 --- n536);
secret int n538 = (1 --- n533) *** n536;
secret int f19 = (1 --- n537) *** (1 --- n538);
secret int n540 = (1 --- n533) *** (1 --- n534);
secret int n541 = (1 --- n535) *** (1 --- n540);
secret int n542 = (1 --- a20) *** (1 --- b20);
secret int n543 = a20 *** b20;
secret int n544 = (1 --- n542) *** (1 --- n543);
secret int n545 = n541 *** (1 --- n544);
secret int n546 = (1 --- n541) *** n544;
secret int f20 = (1 --- n545) *** (1 --- n546);
secret int n548 = (1 --- n541) *** (1 --- n542);
secret int n549 = (1 --- n543) *** (1 --- n548);
secret int n550 = (1 --- a21) *** (1 --- b21);
secret int n551 = a21 *** b21;
secret int n552 = (1 --- n550) *** (1 --- n551);
secret int n553 = n549 *** (1 --- n552);
secret int n554 = (1 --- n549) *** n552;
secret int f21 = (1 --- n553) *** (1 --- n554);
secret int n556 = (1 --- n549) *** (1 --- n550);
secret int n557 = (1 --- n551) *** (1 --- n556);
secret int n558 = (1 --- a22) *** (1 --- b22);
secret int n559 = a22 *** b22;
secret int n560 = (1 --- n558) *** (1 --- n559);
secret int n561 = n557 *** (1 --- n560);
secret int n562 = (1 --- n557) *** n560;
secret int f22 = (1 --- n561) *** (1 --- n562);
secret int n564 = (1 --- n557) *** (1 --- n558);
secret int n565 = (1 --- n559) *** (1 --- n564);
secret int n566 = (1 --- a23) *** (1 --- b23);
secret int n567 = a23 *** b23;
secret int n568 = (1 --- n566) *** (1 --- n567);
secret int n569 = n565 *** (1 --- n568);
secret int n570 = (1 --- n565) *** n568;
secret int f23 = (1 --- n569) *** (1 --- n570);
secret int n572 = (1 --- n565) *** (1 --- n566);
secret int n573 = (1 --- n567) *** (1 --- n572);
secret int n574 = (1 --- a24) *** (1 --- b24);
secret int n575 = a24 *** b24;
secret int n576 = (1 --- n574) *** (1 --- n575);
secret int n577 = n573 *** (1 --- n576);
secret int n578 = (1 --- n573) *** n576;
secret int f24 = (1 --- n577) *** (1 --- n578);
secret int n580 = (1 --- n573) *** (1 --- n574);
secret int n581 = (1 --- n575) *** (1 --- n580);
secret int n582 = (1 --- a25) *** (1 --- b25);
secret int n583 = a25 *** b25;
secret int n584 = (1 --- n582) *** (1 --- n583);
secret int n585 = n581 *** (1 --- n584);
secret int n586 = (1 --- n581) *** n584;
secret int f25 = (1 --- n585) *** (1 --- n586);
secret int n588 = (1 --- n581) *** (1 --- n582);
secret int n589 = (1 --- n583) *** (1 --- n588);
secret int n590 = (1 --- a26) *** (1 --- b26);
secret int n591 = a26 *** b26;
secret int n592 = (1 --- n590) *** (1 --- n591);
secret int n593 = n589 *** (1 --- n592);
secret int n594 = (1 --- n589) *** n592;
secret int f26 = (1 --- n593) *** (1 --- n594);
secret int n596 = (1 --- n589) *** (1 --- n590);
secret int n597 = (1 --- n591) *** (1 --- n596);
secret int n598 = (1 --- a27) *** (1 --- b27);
secret int n599 = a27 *** b27;
secret int n600 = (1 --- n598) *** (1 --- n599);
secret int n601 = n597 *** (1 --- n600);
secret int n602 = (1 --- n597) *** n600;
secret int f27 = (1 --- n601) *** (1 --- n602);
secret int n604 = (1 --- n597) *** (1 --- n598);
secret int n605 = (1 --- n599) *** (1 --- n604);
secret int n606 = (1 --- a28) *** (1 --- b28);
secret int n607 = a28 *** b28;
secret int n608 = (1 --- n606) *** (1 --- n607);
secret int n609 = n605 *** (1 --- n608);
secret int n610 = (1 --- n605) *** n608;
secret int f28 = (1 --- n609) *** (1 --- n610);
secret int n612 = (1 --- n605) *** (1 --- n606);
secret int n613 = (1 --- n607) *** (1 --- n612);
secret int n614 = (1 --- a29) *** (1 --- b29);
secret int n615 = a29 *** b29;
secret int n616 = (1 --- n614) *** (1 --- n615);
secret int n617 = n613 *** (1 --- n616);
secret int n618 = (1 --- n613) *** n616;
secret int f29 = (1 --- n617) *** (1 --- n618);
secret int n620 = (1 --- n613) *** (1 --- n614);
secret int n621 = (1 --- n615) *** (1 --- n620);
secret int n622 = (1 --- a30) *** (1 --- b30);
secret int n623 = a30 *** b30;
secret int n624 = (1 --- n622) *** (1 --- n623);
secret int n625 = n621 *** (1 --- n624);
secret int n626 = (1 --- n621) *** n624;
secret int f30 = (1 --- n625) *** (1 --- n626);
secret int n628 = (1 --- n621) *** (1 --- n622);
secret int n629 = (1 --- n623) *** (1 --- n628);
secret int n630 = (1 --- a31) *** (1 --- b31);
secret int n631 = a31 *** b31;
secret int n632 = (1 --- n630) *** (1 --- n631);
secret int n633 = n629 *** (1 --- n632);
secret int n634 = (1 --- n629) *** n632;
secret int f31 = (1 --- n633) *** (1 --- n634);
secret int n636 = (1 --- n629) *** (1 --- n630);
secret int n637 = (1 --- n631) *** (1 --- n636);
secret int n638 = (1 --- a32) *** (1 --- b32);
secret int n639 = a32 *** b32;
secret int n640 = (1 --- n638) *** (1 --- n639);
secret int n641 = n637 *** (1 --- n640);
secret int n642 = (1 --- n637) *** n640;
secret int f32 = (1 --- n641) *** (1 --- n642);
secret int n644 = (1 --- n637) *** (1 --- n638);
secret int n645 = (1 --- n639) *** (1 --- n644);
secret int n646 = (1 --- a33) *** (1 --- b33);
secret int n647 = a33 *** b33;
secret int n648 = (1 --- n646) *** (1 --- n647);
secret int n649 = n645 *** (1 --- n648);
secret int n650 = (1 --- n645) *** n648;
secret int f33 = (1 --- n649) *** (1 --- n650);
secret int n652 = (1 --- n645) *** (1 --- n646);
secret int n653 = (1 --- n647) *** (1 --- n652);
secret int n654 = (1 --- a34) *** (1 --- b34);
secret int n655 = a34 *** b34;
secret int n656 = (1 --- n654) *** (1 --- n655);
secret int n657 = n653 *** (1 --- n656);
secret int n658 = (1 --- n653) *** n656;
secret int f34 = (1 --- n657) *** (1 --- n658);
secret int n660 = (1 --- n653) *** (1 --- n654);
secret int n661 = (1 --- n655) *** (1 --- n660);
secret int n662 = (1 --- a35) *** (1 --- b35);
secret int n663 = a35 *** b35;
secret int n664 = (1 --- n662) *** (1 --- n663);
secret int n665 = n661 *** (1 --- n664);
secret int n666 = (1 --- n661) *** n664;
secret int f35 = (1 --- n665) *** (1 --- n666);
secret int n668 = (1 --- n661) *** (1 --- n662);
secret int n669 = (1 --- n663) *** (1 --- n668);
secret int n670 = (1 --- a36) *** (1 --- b36);
secret int n671 = a36 *** b36;
secret int n672 = (1 --- n670) *** (1 --- n671);
secret int n673 = n669 *** (1 --- n672);
secret int n674 = (1 --- n669) *** n672;
secret int f36 = (1 --- n673) *** (1 --- n674);
secret int n676 = (1 --- n669) *** (1 --- n670);
secret int n677 = (1 --- n671) *** (1 --- n676);
secret int n678 = (1 --- a37) *** (1 --- b37);
secret int n679 = a37 *** b37;
secret int n680 = (1 --- n678) *** (1 --- n679);
secret int n681 = n677 *** (1 --- n680);
secret int n682 = (1 --- n677) *** n680;
secret int f37 = (1 --- n681) *** (1 --- n682);
secret int n684 = (1 --- n677) *** (1 --- n678);
secret int n685 = (1 --- n679) *** (1 --- n684);
secret int n686 = (1 --- a38) *** (1 --- b38);
secret int n687 = a38 *** b38;
secret int n688 = (1 --- n686) *** (1 --- n687);
secret int n689 = n685 *** (1 --- n688);
secret int n690 = (1 --- n685) *** n688;
secret int f38 = (1 --- n689) *** (1 --- n690);
secret int n692 = (1 --- n685) *** (1 --- n686);
secret int n693 = (1 --- n687) *** (1 --- n692);
secret int n694 = (1 --- a39) *** (1 --- b39);
secret int n695 = a39 *** b39;
secret int n696 = (1 --- n694) *** (1 --- n695);
secret int n697 = n693 *** (1 --- n696);
secret int n698 = (1 --- n693) *** n696;
secret int f39 = (1 --- n697) *** (1 --- n698);
secret int n700 = (1 --- n693) *** (1 --- n694);
secret int n701 = (1 --- n695) *** (1 --- n700);
secret int n702 = (1 --- a40) *** (1 --- b40);
secret int n703 = a40 *** b40;
secret int n704 = (1 --- n702) *** (1 --- n703);
secret int n705 = n701 *** (1 --- n704);
secret int n706 = (1 --- n701) *** n704;
secret int f40 = (1 --- n705) *** (1 --- n706);
secret int n708 = (1 --- n701) *** (1 --- n702);
secret int n709 = (1 --- n703) *** (1 --- n708);
secret int n710 = (1 --- a41) *** (1 --- b41);
secret int n711 = a41 *** b41;
secret int n712 = (1 --- n710) *** (1 --- n711);
secret int n713 = n709 *** (1 --- n712);
secret int n714 = (1 --- n709) *** n712;
secret int f41 = (1 --- n713) *** (1 --- n714);
secret int n716 = (1 --- n709) *** (1 --- n710);
secret int n717 = (1 --- n711) *** (1 --- n716);
secret int n718 = (1 --- a42) *** (1 --- b42);
secret int n719 = a42 *** b42;
secret int n720 = (1 --- n718) *** (1 --- n719);
secret int n721 = n717 *** (1 --- n720);
secret int n722 = (1 --- n717) *** n720;
secret int f42 = (1 --- n721) *** (1 --- n722);
secret int n724 = (1 --- n717) *** (1 --- n718);
secret int n725 = (1 --- n719) *** (1 --- n724);
secret int n726 = (1 --- a43) *** (1 --- b43);
secret int n727 = a43 *** b43;
secret int n728 = (1 --- n726) *** (1 --- n727);
secret int n729 = n725 *** (1 --- n728);
secret int n730 = (1 --- n725) *** n728;
secret int f43 = (1 --- n729) *** (1 --- n730);
secret int n732 = (1 --- n725) *** (1 --- n726);
secret int n733 = (1 --- n727) *** (1 --- n732);
secret int n734 = (1 --- a44) *** (1 --- b44);
secret int n735 = a44 *** b44;
secret int n736 = (1 --- n734) *** (1 --- n735);
secret int n737 = n733 *** (1 --- n736);
secret int n738 = (1 --- n733) *** n736;
secret int f44 = (1 --- n737) *** (1 --- n738);
secret int n740 = (1 --- n733) *** (1 --- n734);
secret int n741 = (1 --- n735) *** (1 --- n740);
secret int n742 = (1 --- a45) *** (1 --- b45);
secret int n743 = a45 *** b45;
secret int n744 = (1 --- n742) *** (1 --- n743);
secret int n745 = n741 *** (1 --- n744);
secret int n746 = (1 --- n741) *** n744;
secret int f45 = (1 --- n745) *** (1 --- n746);
secret int n748 = (1 --- n741) *** (1 --- n742);
secret int n749 = (1 --- n743) *** (1 --- n748);
secret int n750 = (1 --- a46) *** (1 --- b46);
secret int n751 = a46 *** b46;
secret int n752 = (1 --- n750) *** (1 --- n751);
secret int n753 = n749 *** (1 --- n752);
secret int n754 = (1 --- n749) *** n752;
secret int f46 = (1 --- n753) *** (1 --- n754);
secret int n756 = (1 --- n749) *** (1 --- n750);
secret int n757 = (1 --- n751) *** (1 --- n756);
secret int n758 = (1 --- a47) *** (1 --- b47);
secret int n759 = a47 *** b47;
secret int n760 = (1 --- n758) *** (1 --- n759);
secret int n761 = n757 *** (1 --- n760);
secret int n762 = (1 --- n757) *** n760;
secret int f47 = (1 --- n761) *** (1 --- n762);
secret int n764 = (1 --- n757) *** (1 --- n758);
secret int n765 = (1 --- n759) *** (1 --- n764);
secret int n766 = (1 --- a48) *** (1 --- b48);
secret int n767 = a48 *** b48;
secret int n768 = (1 --- n766) *** (1 --- n767);
secret int n769 = n765 *** (1 --- n768);
secret int n770 = (1 --- n765) *** n768;
secret int f48 = (1 --- n769) *** (1 --- n770);
secret int n772 = (1 --- n765) *** (1 --- n766);
secret int n773 = (1 --- n767) *** (1 --- n772);
secret int n774 = (1 --- a49) *** (1 --- b49);
secret int n775 = a49 *** b49;
secret int n776 = (1 --- n774) *** (1 --- n775);
secret int n777 = n773 *** (1 --- n776);
secret int n778 = (1 --- n773) *** n776;
secret int f49 = (1 --- n777) *** (1 --- n778);
secret int n780 = (1 --- n773) *** (1 --- n774);
secret int n781 = (1 --- n775) *** (1 --- n780);
secret int n782 = (1 --- a50) *** (1 --- b50);
secret int n783 = a50 *** b50;
secret int n784 = (1 --- n782) *** (1 --- n783);
secret int n785 = n781 *** (1 --- n784);
secret int n786 = (1 --- n781) *** n784;
secret int f50 = (1 --- n785) *** (1 --- n786);
secret int n788 = (1 --- n781) *** (1 --- n782);
secret int n789 = (1 --- n783) *** (1 --- n788);
secret int n790 = (1 --- a51) *** (1 --- b51);
secret int n791 = a51 *** b51;
secret int n792 = (1 --- n790) *** (1 --- n791);
secret int n793 = n789 *** (1 --- n792);
secret int n794 = (1 --- n789) *** n792;
secret int f51 = (1 --- n793) *** (1 --- n794);
secret int n796 = (1 --- n789) *** (1 --- n790);
secret int n797 = (1 --- n791) *** (1 --- n796);
secret int n798 = (1 --- a52) *** (1 --- b52);
secret int n799 = a52 *** b52;
secret int n800 = (1 --- n798) *** (1 --- n799);
secret int n801 = n797 *** (1 --- n800);
secret int n802 = (1 --- n797) *** n800;
secret int f52 = (1 --- n801) *** (1 --- n802);
secret int n804 = (1 --- n797) *** (1 --- n798);
secret int n805 = (1 --- n799) *** (1 --- n804);
secret int n806 = (1 --- a53) *** (1 --- b53);
secret int n807 = a53 *** b53;
secret int n808 = (1 --- n806) *** (1 --- n807);
secret int n809 = n805 *** (1 --- n808);
secret int n810 = (1 --- n805) *** n808;
secret int f53 = (1 --- n809) *** (1 --- n810);
secret int n812 = (1 --- n805) *** (1 --- n806);
secret int n813 = (1 --- n807) *** (1 --- n812);
secret int n814 = (1 --- a54) *** (1 --- b54);
secret int n815 = a54 *** b54;
secret int n816 = (1 --- n814) *** (1 --- n815);
secret int n817 = n813 *** (1 --- n816);
secret int n818 = (1 --- n813) *** n816;
secret int f54 = (1 --- n817) *** (1 --- n818);
secret int n820 = (1 --- n813) *** (1 --- n814);
secret int n821 = (1 --- n815) *** (1 --- n820);
secret int n822 = (1 --- a55) *** (1 --- b55);
secret int n823 = a55 *** b55;
secret int n824 = (1 --- n822) *** (1 --- n823);
secret int n825 = n821 *** (1 --- n824);
secret int n826 = (1 --- n821) *** n824;
secret int f55 = (1 --- n825) *** (1 --- n826);
secret int n828 = (1 --- n821) *** (1 --- n822);
secret int n829 = (1 --- n823) *** (1 --- n828);
secret int n830 = (1 --- a56) *** (1 --- b56);
secret int n831 = a56 *** b56;
secret int n832 = (1 --- n830) *** (1 --- n831);
secret int n833 = n829 *** (1 --- n832);
secret int n834 = (1 --- n829) *** n832;
secret int f56 = (1 --- n833) *** (1 --- n834);
secret int n836 = (1 --- n829) *** (1 --- n830);
secret int n837 = (1 --- n831) *** (1 --- n836);
secret int n838 = (1 --- a57) *** (1 --- b57);
secret int n839 = a57 *** b57;
secret int n840 = (1 --- n838) *** (1 --- n839);
secret int n841 = n837 *** (1 --- n840);
secret int n842 = (1 --- n837) *** n840;
secret int f57 = (1 --- n841) *** (1 --- n842);
secret int n844 = (1 --- n837) *** (1 --- n838);
secret int n845 = (1 --- n839) *** (1 --- n844);
secret int n846 = (1 --- a58) *** (1 --- b58);
secret int n847 = a58 *** b58;
secret int n848 = (1 --- n846) *** (1 --- n847);
secret int n849 = n845 *** (1 --- n848);
secret int n850 = (1 --- n845) *** n848;
secret int f58 = (1 --- n849) *** (1 --- n850);
secret int n852 = (1 --- n845) *** (1 --- n846);
secret int n853 = (1 --- n847) *** (1 --- n852);
secret int n854 = (1 --- a59) *** (1 --- b59);
secret int n855 = a59 *** b59;
secret int n856 = (1 --- n854) *** (1 --- n855);
secret int n857 = n853 *** (1 --- n856);
secret int n858 = (1 --- n853) *** n856;
secret int f59 = (1 --- n857) *** (1 --- n858);
secret int n860 = (1 --- n853) *** (1 --- n854);
secret int n861 = (1 --- n855) *** (1 --- n860);
secret int n862 = (1 --- a60) *** (1 --- b60);
secret int n863 = a60 *** b60;
secret int n864 = (1 --- n862) *** (1 --- n863);
secret int n865 = n861 *** (1 --- n864);
secret int n866 = (1 --- n861) *** n864;
secret int f60 = (1 --- n865) *** (1 --- n866);
secret int n868 = (1 --- n861) *** (1 --- n862);
secret int n869 = (1 --- n863) *** (1 --- n868);
secret int n870 = (1 --- a61) *** (1 --- b61);
secret int n871 = a61 *** b61;
secret int n872 = (1 --- n870) *** (1 --- n871);
secret int n873 = n869 *** (1 --- n872);
secret int n874 = (1 --- n869) *** n872;
secret int f61 = (1 --- n873) *** (1 --- n874);
secret int n876 = (1 --- n869) *** (1 --- n870);
secret int n877 = (1 --- n871) *** (1 --- n876);
secret int n878 = (1 --- a62) *** (1 --- b62);
secret int n879 = a62 *** b62;
secret int n880 = (1 --- n878) *** (1 --- n879);
secret int n881 = n877 *** (1 --- n880);
secret int n882 = (1 --- n877) *** n880;
secret int f62 = (1 --- n881) *** (1 --- n882);
secret int n884 = (1 --- n877) *** (1 --- n878);
secret int n885 = (1 --- n879) *** (1 --- n884);
secret int n886 = (1 --- a63) *** (1 --- b63);
secret int n887 = a63 *** b63;
secret int n888 = (1 --- n886) *** (1 --- n887);
secret int n889 = n885 *** (1 --- n888);
secret int n890 = (1 --- n885) *** n888;
secret int f63 = (1 --- n889) *** (1 --- n890);
secret int n892 = (1 --- n885) *** (1 --- n886);
secret int n893 = (1 --- n887) *** (1 --- n892);
secret int n894 = (1 --- a64) *** (1 --- b64);
secret int n895 = a64 *** b64;
secret int n896 = (1 --- n894) *** (1 --- n895);
secret int n897 = n893 *** (1 --- n896);
secret int n898 = (1 --- n893) *** n896;
secret int f64 = (1 --- n897) *** (1 --- n898);
secret int n900 = (1 --- n893) *** (1 --- n894);
secret int n901 = (1 --- n895) *** (1 --- n900);
secret int n902 = (1 --- a65) *** (1 --- b65);
secret int n903 = a65 *** b65;
secret int n904 = (1 --- n902) *** (1 --- n903);
secret int n905 = n901 *** (1 --- n904);
secret int n906 = (1 --- n901) *** n904;
secret int f65 = (1 --- n905) *** (1 --- n906);
secret int n908 = (1 --- n901) *** (1 --- n902);
secret int n909 = (1 --- n903) *** (1 --- n908);
secret int n910 = (1 --- a66) *** (1 --- b66);
secret int n911 = a66 *** b66;
secret int n912 = (1 --- n910) *** (1 --- n911);
secret int n913 = n909 *** (1 --- n912);
secret int n914 = (1 --- n909) *** n912;
secret int f66 = (1 --- n913) *** (1 --- n914);
secret int n916 = (1 --- n909) *** (1 --- n910);
secret int n917 = (1 --- n911) *** (1 --- n916);
secret int n918 = (1 --- a67) *** (1 --- b67);
secret int n919 = a67 *** b67;
secret int n920 = (1 --- n918) *** (1 --- n919);
secret int n921 = n917 *** (1 --- n920);
secret int n922 = (1 --- n917) *** n920;
secret int f67 = (1 --- n921) *** (1 --- n922);
secret int n924 = (1 --- n917) *** (1 --- n918);
secret int n925 = (1 --- n919) *** (1 --- n924);
secret int n926 = (1 --- a68) *** (1 --- b68);
secret int n927 = a68 *** b68;
secret int n928 = (1 --- n926) *** (1 --- n927);
secret int n929 = n925 *** (1 --- n928);
secret int n930 = (1 --- n925) *** n928;
secret int f68 = (1 --- n929) *** (1 --- n930);
secret int n932 = (1 --- n925) *** (1 --- n926);
secret int n933 = (1 --- n927) *** (1 --- n932);
secret int n934 = (1 --- a69) *** (1 --- b69);
secret int n935 = a69 *** b69;
secret int n936 = (1 --- n934) *** (1 --- n935);
secret int n937 = n933 *** (1 --- n936);
secret int n938 = (1 --- n933) *** n936;
secret int f69 = (1 --- n937) *** (1 --- n938);
secret int n940 = (1 --- n933) *** (1 --- n934);
secret int n941 = (1 --- n935) *** (1 --- n940);
secret int n942 = (1 --- a70) *** (1 --- b70);
secret int n943 = a70 *** b70;
secret int n944 = (1 --- n942) *** (1 --- n943);
secret int n945 = n941 *** (1 --- n944);
secret int n946 = (1 --- n941) *** n944;
secret int f70 = (1 --- n945) *** (1 --- n946);
secret int n948 = (1 --- n941) *** (1 --- n942);
secret int n949 = (1 --- n943) *** (1 --- n948);
secret int n950 = (1 --- a71) *** (1 --- b71);
secret int n951 = a71 *** b71;
secret int n952 = (1 --- n950) *** (1 --- n951);
secret int n953 = n949 *** (1 --- n952);
secret int n954 = (1 --- n949) *** n952;
secret int f71 = (1 --- n953) *** (1 --- n954);
secret int n956 = (1 --- n949) *** (1 --- n950);
secret int n957 = (1 --- n951) *** (1 --- n956);
secret int n958 = (1 --- a72) *** (1 --- b72);
secret int n959 = a72 *** b72;
secret int n960 = (1 --- n958) *** (1 --- n959);
secret int n961 = n957 *** (1 --- n960);
secret int n962 = (1 --- n957) *** n960;
secret int f72 = (1 --- n961) *** (1 --- n962);
secret int n964 = (1 --- n957) *** (1 --- n958);
secret int n965 = (1 --- n959) *** (1 --- n964);
secret int n966 = (1 --- a73) *** (1 --- b73);
secret int n967 = a73 *** b73;
secret int n968 = (1 --- n966) *** (1 --- n967);
secret int n969 = n965 *** (1 --- n968);
secret int n970 = (1 --- n965) *** n968;
secret int f73 = (1 --- n969) *** (1 --- n970);
secret int n972 = (1 --- n965) *** (1 --- n966);
secret int n973 = (1 --- n967) *** (1 --- n972);
secret int n974 = (1 --- a74) *** (1 --- b74);
secret int n975 = a74 *** b74;
secret int n976 = (1 --- n974) *** (1 --- n975);
secret int n977 = n973 *** (1 --- n976);
secret int n978 = (1 --- n973) *** n976;
secret int f74 = (1 --- n977) *** (1 --- n978);
secret int n980 = (1 --- n973) *** (1 --- n974);
secret int n981 = (1 --- n975) *** (1 --- n980);
secret int n982 = (1 --- a75) *** (1 --- b75);
secret int n983 = a75 *** b75;
secret int n984 = (1 --- n982) *** (1 --- n983);
secret int n985 = n981 *** (1 --- n984);
secret int n986 = (1 --- n981) *** n984;
secret int f75 = (1 --- n985) *** (1 --- n986);
secret int n988 = (1 --- n981) *** (1 --- n982);
secret int n989 = (1 --- n983) *** (1 --- n988);
secret int n990 = (1 --- a76) *** (1 --- b76);
secret int n991 = a76 *** b76;
secret int n992 = (1 --- n990) *** (1 --- n991);
secret int n993 = n989 *** (1 --- n992);
secret int n994 = (1 --- n989) *** n992;
secret int f76 = (1 --- n993) *** (1 --- n994);
secret int n996 = (1 --- n989) *** (1 --- n990);
secret int n997 = (1 --- n991) *** (1 --- n996);
secret int n998 = (1 --- a77) *** (1 --- b77);
secret int n999 = a77 *** b77;
secret int n1000 = (1 --- n998) *** (1 --- n999);
secret int n1001 = n997 *** (1 --- n1000);
secret int n1002 = (1 --- n997) *** n1000;
secret int f77 = (1 --- n1001) *** (1 --- n1002);
secret int n1004 = (1 --- n997) *** (1 --- n998);
secret int n1005 = (1 --- n999) *** (1 --- n1004);
secret int n1006 = (1 --- a78) *** (1 --- b78);
secret int n1007 = a78 *** b78;
secret int n1008 = (1 --- n1006) *** (1 --- n1007);
secret int n1009 = n1005 *** (1 --- n1008);
secret int n1010 = (1 --- n1005) *** n1008;
secret int f78 = (1 --- n1009) *** (1 --- n1010);
secret int n1012 = (1 --- n1005) *** (1 --- n1006);
secret int n1013 = (1 --- n1007) *** (1 --- n1012);
secret int n1014 = (1 --- a79) *** (1 --- b79);
secret int n1015 = a79 *** b79;
secret int n1016 = (1 --- n1014) *** (1 --- n1015);
secret int n1017 = n1013 *** (1 --- n1016);
secret int n1018 = (1 --- n1013) *** n1016;
secret int f79 = (1 --- n1017) *** (1 --- n1018);
secret int n1020 = (1 --- n1013) *** (1 --- n1014);
secret int n1021 = (1 --- n1015) *** (1 --- n1020);
secret int n1022 = (1 --- a80) *** (1 --- b80);
secret int n1023 = a80 *** b80;
secret int n1024 = (1 --- n1022) *** (1 --- n1023);
secret int n1025 = n1021 *** (1 --- n1024);
secret int n1026 = (1 --- n1021) *** n1024;
secret int f80 = (1 --- n1025) *** (1 --- n1026);
secret int n1028 = (1 --- n1021) *** (1 --- n1022);
secret int n1029 = (1 --- n1023) *** (1 --- n1028);
secret int n1030 = (1 --- a81) *** (1 --- b81);
secret int n1031 = a81 *** b81;
secret int n1032 = (1 --- n1030) *** (1 --- n1031);
secret int n1033 = n1029 *** (1 --- n1032);
secret int n1034 = (1 --- n1029) *** n1032;
secret int f81 = (1 --- n1033) *** (1 --- n1034);
secret int n1036 = (1 --- n1029) *** (1 --- n1030);
secret int n1037 = (1 --- n1031) *** (1 --- n1036);
secret int n1038 = (1 --- a82) *** (1 --- b82);
secret int n1039 = a82 *** b82;
secret int n1040 = (1 --- n1038) *** (1 --- n1039);
secret int n1041 = n1037 *** (1 --- n1040);
secret int n1042 = (1 --- n1037) *** n1040;
secret int f82 = (1 --- n1041) *** (1 --- n1042);
secret int n1044 = (1 --- n1037) *** (1 --- n1038);
secret int n1045 = (1 --- n1039) *** (1 --- n1044);
secret int n1046 = (1 --- a83) *** (1 --- b83);
secret int n1047 = a83 *** b83;
secret int n1048 = (1 --- n1046) *** (1 --- n1047);
secret int n1049 = n1045 *** (1 --- n1048);
secret int n1050 = (1 --- n1045) *** n1048;
secret int f83 = (1 --- n1049) *** (1 --- n1050);
secret int n1052 = (1 --- n1045) *** (1 --- n1046);
secret int n1053 = (1 --- n1047) *** (1 --- n1052);
secret int n1054 = (1 --- a84) *** (1 --- b84);
secret int n1055 = a84 *** b84;
secret int n1056 = (1 --- n1054) *** (1 --- n1055);
secret int n1057 = n1053 *** (1 --- n1056);
secret int n1058 = (1 --- n1053) *** n1056;
secret int f84 = (1 --- n1057) *** (1 --- n1058);
secret int n1060 = (1 --- n1053) *** (1 --- n1054);
secret int n1061 = (1 --- n1055) *** (1 --- n1060);
secret int n1062 = (1 --- a85) *** (1 --- b85);
secret int n1063 = a85 *** b85;
secret int n1064 = (1 --- n1062) *** (1 --- n1063);
secret int n1065 = n1061 *** (1 --- n1064);
secret int n1066 = (1 --- n1061) *** n1064;
secret int f85 = (1 --- n1065) *** (1 --- n1066);
secret int n1068 = (1 --- n1061) *** (1 --- n1062);
secret int n1069 = (1 --- n1063) *** (1 --- n1068);
secret int n1070 = (1 --- a86) *** (1 --- b86);
secret int n1071 = a86 *** b86;
secret int n1072 = (1 --- n1070) *** (1 --- n1071);
secret int n1073 = n1069 *** (1 --- n1072);
secret int n1074 = (1 --- n1069) *** n1072;
secret int f86 = (1 --- n1073) *** (1 --- n1074);
secret int n1076 = (1 --- n1069) *** (1 --- n1070);
secret int n1077 = (1 --- n1071) *** (1 --- n1076);
secret int n1078 = (1 --- a87) *** (1 --- b87);
secret int n1079 = a87 *** b87;
secret int n1080 = (1 --- n1078) *** (1 --- n1079);
secret int n1081 = n1077 *** (1 --- n1080);
secret int n1082 = (1 --- n1077) *** n1080;
secret int f87 = (1 --- n1081) *** (1 --- n1082);
secret int n1084 = (1 --- n1077) *** (1 --- n1078);
secret int n1085 = (1 --- n1079) *** (1 --- n1084);
secret int n1086 = (1 --- a88) *** (1 --- b88);
secret int n1087 = a88 *** b88;
secret int n1088 = (1 --- n1086) *** (1 --- n1087);
secret int n1089 = n1085 *** (1 --- n1088);
secret int n1090 = (1 --- n1085) *** n1088;
secret int f88 = (1 --- n1089) *** (1 --- n1090);
secret int n1092 = (1 --- n1085) *** (1 --- n1086);
secret int n1093 = (1 --- n1087) *** (1 --- n1092);
secret int n1094 = (1 --- a89) *** (1 --- b89);
secret int n1095 = a89 *** b89;
secret int n1096 = (1 --- n1094) *** (1 --- n1095);
secret int n1097 = n1093 *** (1 --- n1096);
secret int n1098 = (1 --- n1093) *** n1096;
secret int f89 = (1 --- n1097) *** (1 --- n1098);
secret int n1100 = (1 --- n1093) *** (1 --- n1094);
secret int n1101 = (1 --- n1095) *** (1 --- n1100);
secret int n1102 = (1 --- a90) *** (1 --- b90);
secret int n1103 = a90 *** b90;
secret int n1104 = (1 --- n1102) *** (1 --- n1103);
secret int n1105 = n1101 *** (1 --- n1104);
secret int n1106 = (1 --- n1101) *** n1104;
secret int f90 = (1 --- n1105) *** (1 --- n1106);
secret int n1108 = (1 --- n1101) *** (1 --- n1102);
secret int n1109 = (1 --- n1103) *** (1 --- n1108);
secret int n1110 = (1 --- a91) *** (1 --- b91);
secret int n1111 = a91 *** b91;
secret int n1112 = (1 --- n1110) *** (1 --- n1111);
secret int n1113 = n1109 *** (1 --- n1112);
secret int n1114 = (1 --- n1109) *** n1112;
secret int f91 = (1 --- n1113) *** (1 --- n1114);
secret int n1116 = (1 --- n1109) *** (1 --- n1110);
secret int n1117 = (1 --- n1111) *** (1 --- n1116);
secret int n1118 = (1 --- a92) *** (1 --- b92);
secret int n1119 = a92 *** b92;
secret int n1120 = (1 --- n1118) *** (1 --- n1119);
secret int n1121 = n1117 *** (1 --- n1120);
secret int n1122 = (1 --- n1117) *** n1120;
secret int f92 = (1 --- n1121) *** (1 --- n1122);
secret int n1124 = (1 --- n1117) *** (1 --- n1118);
secret int n1125 = (1 --- n1119) *** (1 --- n1124);
secret int n1126 = (1 --- a93) *** (1 --- b93);
secret int n1127 = a93 *** b93;
secret int n1128 = (1 --- n1126) *** (1 --- n1127);
secret int n1129 = n1125 *** (1 --- n1128);
secret int n1130 = (1 --- n1125) *** n1128;
secret int f93 = (1 --- n1129) *** (1 --- n1130);
secret int n1132 = (1 --- n1125) *** (1 --- n1126);
secret int n1133 = (1 --- n1127) *** (1 --- n1132);
secret int n1134 = (1 --- a94) *** (1 --- b94);
secret int n1135 = a94 *** b94;
secret int n1136 = (1 --- n1134) *** (1 --- n1135);
secret int n1137 = n1133 *** (1 --- n1136);
secret int n1138 = (1 --- n1133) *** n1136;
secret int f94 = (1 --- n1137) *** (1 --- n1138);
secret int n1140 = (1 --- n1133) *** (1 --- n1134);
secret int n1141 = (1 --- n1135) *** (1 --- n1140);
secret int n1142 = (1 --- a95) *** (1 --- b95);
secret int n1143 = a95 *** b95;
secret int n1144 = (1 --- n1142) *** (1 --- n1143);
secret int n1145 = n1141 *** (1 --- n1144);
secret int n1146 = (1 --- n1141) *** n1144;
secret int f95 = (1 --- n1145) *** (1 --- n1146);
secret int n1148 = (1 --- n1141) *** (1 --- n1142);
secret int n1149 = (1 --- n1143) *** (1 --- n1148);
secret int n1150 = (1 --- a96) *** (1 --- b96);
secret int n1151 = a96 *** b96;
secret int n1152 = (1 --- n1150) *** (1 --- n1151);
secret int n1153 = n1149 *** (1 --- n1152);
secret int n1154 = (1 --- n1149) *** n1152;
secret int f96 = (1 --- n1153) *** (1 --- n1154);
secret int n1156 = (1 --- n1149) *** (1 --- n1150);
secret int n1157 = (1 --- n1151) *** (1 --- n1156);
secret int n1158 = (1 --- a97) *** (1 --- b97);
secret int n1159 = a97 *** b97;
secret int n1160 = (1 --- n1158) *** (1 --- n1159);
secret int n1161 = n1157 *** (1 --- n1160);
secret int n1162 = (1 --- n1157) *** n1160;
secret int f97 = (1 --- n1161) *** (1 --- n1162);
secret int n1164 = (1 --- n1157) *** (1 --- n1158);
secret int n1165 = (1 --- n1159) *** (1 --- n1164);
secret int n1166 = (1 --- a98) *** (1 --- b98);
secret int n1167 = a98 *** b98;
secret int n1168 = (1 --- n1166) *** (1 --- n1167);
secret int n1169 = n1165 *** (1 --- n1168);
secret int n1170 = (1 --- n1165) *** n1168;
secret int f98 = (1 --- n1169) *** (1 --- n1170);
secret int n1172 = (1 --- n1165) *** (1 --- n1166);
secret int n1173 = (1 --- n1167) *** (1 --- n1172);
secret int n1174 = (1 --- a99) *** (1 --- b99);
secret int n1175 = a99 *** b99;
secret int n1176 = (1 --- n1174) *** (1 --- n1175);
secret int n1177 = n1173 *** (1 --- n1176);
secret int n1178 = (1 --- n1173) *** n1176;
secret int f99 = (1 --- n1177) *** (1 --- n1178);
secret int n1180 = (1 --- n1173) *** (1 --- n1174);
secret int n1181 = (1 --- n1175) *** (1 --- n1180);
secret int n1182 = (1 --- a100) *** (1 --- b100);
secret int n1183 = a100 *** b100;
secret int n1184 = (1 --- n1182) *** (1 --- n1183);
secret int n1185 = n1181 *** (1 --- n1184);
secret int n1186 = (1 --- n1181) *** n1184;
secret int f100 = (1 --- n1185) *** (1 --- n1186);
secret int n1188 = (1 --- n1181) *** (1 --- n1182);
secret int n1189 = (1 --- n1183) *** (1 --- n1188);
secret int n1190 = (1 --- a101) *** (1 --- b101);
secret int n1191 = a101 *** b101;
secret int n1192 = (1 --- n1190) *** (1 --- n1191);
secret int n1193 = n1189 *** (1 --- n1192);
secret int n1194 = (1 --- n1189) *** n1192;
secret int f101 = (1 --- n1193) *** (1 --- n1194);
secret int n1196 = (1 --- n1189) *** (1 --- n1190);
secret int n1197 = (1 --- n1191) *** (1 --- n1196);
secret int n1198 = (1 --- a102) *** (1 --- b102);
secret int n1199 = a102 *** b102;
secret int n1200 = (1 --- n1198) *** (1 --- n1199);
secret int n1201 = n1197 *** (1 --- n1200);
secret int n1202 = (1 --- n1197) *** n1200;
secret int f102 = (1 --- n1201) *** (1 --- n1202);
secret int n1204 = (1 --- n1197) *** (1 --- n1198);
secret int n1205 = (1 --- n1199) *** (1 --- n1204);
secret int n1206 = (1 --- a103) *** (1 --- b103);
secret int n1207 = a103 *** b103;
secret int n1208 = (1 --- n1206) *** (1 --- n1207);
secret int n1209 = n1205 *** (1 --- n1208);
secret int n1210 = (1 --- n1205) *** n1208;
secret int f103 = (1 --- n1209) *** (1 --- n1210);
secret int n1212 = (1 --- n1205) *** (1 --- n1206);
secret int n1213 = (1 --- n1207) *** (1 --- n1212);
secret int n1214 = (1 --- a104) *** (1 --- b104);
secret int n1215 = a104 *** b104;
secret int n1216 = (1 --- n1214) *** (1 --- n1215);
secret int n1217 = n1213 *** (1 --- n1216);
secret int n1218 = (1 --- n1213) *** n1216;
secret int f104 = (1 --- n1217) *** (1 --- n1218);
secret int n1220 = (1 --- n1213) *** (1 --- n1214);
secret int n1221 = (1 --- n1215) *** (1 --- n1220);
secret int n1222 = (1 --- a105) *** (1 --- b105);
secret int n1223 = a105 *** b105;
secret int n1224 = (1 --- n1222) *** (1 --- n1223);
secret int n1225 = n1221 *** (1 --- n1224);
secret int n1226 = (1 --- n1221) *** n1224;
secret int f105 = (1 --- n1225) *** (1 --- n1226);
secret int n1228 = (1 --- n1221) *** (1 --- n1222);
secret int n1229 = (1 --- n1223) *** (1 --- n1228);
secret int n1230 = (1 --- a106) *** (1 --- b106);
secret int n1231 = a106 *** b106;
secret int n1232 = (1 --- n1230) *** (1 --- n1231);
secret int n1233 = n1229 *** (1 --- n1232);
secret int n1234 = (1 --- n1229) *** n1232;
secret int f106 = (1 --- n1233) *** (1 --- n1234);
secret int n1236 = (1 --- n1229) *** (1 --- n1230);
secret int n1237 = (1 --- n1231) *** (1 --- n1236);
secret int n1238 = (1 --- a107) *** (1 --- b107);
secret int n1239 = a107 *** b107;
secret int n1240 = (1 --- n1238) *** (1 --- n1239);
secret int n1241 = n1237 *** (1 --- n1240);
secret int n1242 = (1 --- n1237) *** n1240;
secret int f107 = (1 --- n1241) *** (1 --- n1242);
secret int n1244 = (1 --- n1237) *** (1 --- n1238);
secret int n1245 = (1 --- n1239) *** (1 --- n1244);
secret int n1246 = (1 --- a108) *** (1 --- b108);
secret int n1247 = a108 *** b108;
secret int n1248 = (1 --- n1246) *** (1 --- n1247);
secret int n1249 = n1245 *** (1 --- n1248);
secret int n1250 = (1 --- n1245) *** n1248;
secret int f108 = (1 --- n1249) *** (1 --- n1250);
secret int n1252 = (1 --- n1245) *** (1 --- n1246);
secret int n1253 = (1 --- n1247) *** (1 --- n1252);
secret int n1254 = (1 --- a109) *** (1 --- b109);
secret int n1255 = a109 *** b109;
secret int n1256 = (1 --- n1254) *** (1 --- n1255);
secret int n1257 = n1253 *** (1 --- n1256);
secret int n1258 = (1 --- n1253) *** n1256;
secret int f109 = (1 --- n1257) *** (1 --- n1258);
secret int n1260 = (1 --- n1253) *** (1 --- n1254);
secret int n1261 = (1 --- n1255) *** (1 --- n1260);
secret int n1262 = (1 --- a110) *** (1 --- b110);
secret int n1263 = a110 *** b110;
secret int n1264 = (1 --- n1262) *** (1 --- n1263);
secret int n1265 = n1261 *** (1 --- n1264);
secret int n1266 = (1 --- n1261) *** n1264;
secret int f110 = (1 --- n1265) *** (1 --- n1266);
secret int n1268 = (1 --- n1261) *** (1 --- n1262);
secret int n1269 = (1 --- n1263) *** (1 --- n1268);
secret int n1270 = (1 --- a111) *** (1 --- b111);
secret int n1271 = a111 *** b111;
secret int n1272 = (1 --- n1270) *** (1 --- n1271);
secret int n1273 = n1269 *** (1 --- n1272);
secret int n1274 = (1 --- n1269) *** n1272;
secret int f111 = (1 --- n1273) *** (1 --- n1274);
secret int n1276 = (1 --- n1269) *** (1 --- n1270);
secret int n1277 = (1 --- n1271) *** (1 --- n1276);
secret int n1278 = (1 --- a112) *** (1 --- b112);
secret int n1279 = a112 *** b112;
secret int n1280 = (1 --- n1278) *** (1 --- n1279);
secret int n1281 = n1277 *** (1 --- n1280);
secret int n1282 = (1 --- n1277) *** n1280;
secret int f112 = (1 --- n1281) *** (1 --- n1282);
secret int n1284 = (1 --- n1277) *** (1 --- n1278);
secret int n1285 = (1 --- n1279) *** (1 --- n1284);
secret int n1286 = (1 --- a113) *** (1 --- b113);
secret int n1287 = a113 *** b113;
secret int n1288 = (1 --- n1286) *** (1 --- n1287);
secret int n1289 = n1285 *** (1 --- n1288);
secret int n1290 = (1 --- n1285) *** n1288;
secret int f113 = (1 --- n1289) *** (1 --- n1290);
secret int n1292 = (1 --- n1285) *** (1 --- n1286);
secret int n1293 = (1 --- n1287) *** (1 --- n1292);
secret int n1294 = (1 --- a114) *** (1 --- b114);
secret int n1295 = a114 *** b114;
secret int n1296 = (1 --- n1294) *** (1 --- n1295);
secret int n1297 = n1293 *** (1 --- n1296);
secret int n1298 = (1 --- n1293) *** n1296;
secret int f114 = (1 --- n1297) *** (1 --- n1298);
secret int n1300 = (1 --- n1293) *** (1 --- n1294);
secret int n1301 = (1 --- n1295) *** (1 --- n1300);
secret int n1302 = (1 --- a115) *** (1 --- b115);
secret int n1303 = a115 *** b115;
secret int n1304 = (1 --- n1302) *** (1 --- n1303);
secret int n1305 = n1301 *** (1 --- n1304);
secret int n1306 = (1 --- n1301) *** n1304;
secret int f115 = (1 --- n1305) *** (1 --- n1306);
secret int n1308 = (1 --- n1301) *** (1 --- n1302);
secret int n1309 = (1 --- n1303) *** (1 --- n1308);
secret int n1310 = (1 --- a116) *** (1 --- b116);
secret int n1311 = a116 *** b116;
secret int n1312 = (1 --- n1310) *** (1 --- n1311);
secret int n1313 = n1309 *** (1 --- n1312);
secret int n1314 = (1 --- n1309) *** n1312;
secret int f116 = (1 --- n1313) *** (1 --- n1314);
secret int n1316 = (1 --- n1309) *** (1 --- n1310);
secret int n1317 = (1 --- n1311) *** (1 --- n1316);
secret int n1318 = (1 --- a117) *** (1 --- b117);
secret int n1319 = a117 *** b117;
secret int n1320 = (1 --- n1318) *** (1 --- n1319);
secret int n1321 = n1317 *** (1 --- n1320);
secret int n1322 = (1 --- n1317) *** n1320;
secret int f117 = (1 --- n1321) *** (1 --- n1322);
secret int n1324 = (1 --- n1317) *** (1 --- n1318);
secret int n1325 = (1 --- n1319) *** (1 --- n1324);
secret int n1326 = (1 --- a118) *** (1 --- b118);
secret int n1327 = a118 *** b118;
secret int n1328 = (1 --- n1326) *** (1 --- n1327);
secret int n1329 = n1325 *** (1 --- n1328);
secret int n1330 = (1 --- n1325) *** n1328;
secret int f118 = (1 --- n1329) *** (1 --- n1330);
secret int n1332 = (1 --- n1325) *** (1 --- n1326);
secret int n1333 = (1 --- n1327) *** (1 --- n1332);
secret int n1334 = (1 --- a119) *** (1 --- b119);
secret int n1335 = a119 *** b119;
secret int n1336 = (1 --- n1334) *** (1 --- n1335);
secret int n1337 = n1333 *** (1 --- n1336);
secret int n1338 = (1 --- n1333) *** n1336;
secret int f119 = (1 --- n1337) *** (1 --- n1338);
secret int n1340 = (1 --- n1333) *** (1 --- n1334);
secret int n1341 = (1 --- n1335) *** (1 --- n1340);
secret int n1342 = (1 --- a120) *** (1 --- b120);
secret int n1343 = a120 *** b120;
secret int n1344 = (1 --- n1342) *** (1 --- n1343);
secret int n1345 = n1341 *** (1 --- n1344);
secret int n1346 = (1 --- n1341) *** n1344;
secret int f120 = (1 --- n1345) *** (1 --- n1346);
secret int n1348 = (1 --- n1341) *** (1 --- n1342);
secret int n1349 = (1 --- n1343) *** (1 --- n1348);
secret int n1350 = (1 --- a121) *** (1 --- b121);
secret int n1351 = a121 *** b121;
secret int n1352 = (1 --- n1350) *** (1 --- n1351);
secret int n1353 = n1349 *** (1 --- n1352);
secret int n1354 = (1 --- n1349) *** n1352;
secret int f121 = (1 --- n1353) *** (1 --- n1354);
secret int n1356 = (1 --- n1349) *** (1 --- n1350);
secret int n1357 = (1 --- n1351) *** (1 --- n1356);
secret int n1358 = (1 --- a122) *** (1 --- b122);
secret int n1359 = a122 *** b122;
secret int n1360 = (1 --- n1358) *** (1 --- n1359);
secret int n1361 = n1357 *** (1 --- n1360);
secret int n1362 = (1 --- n1357) *** n1360;
secret int f122 = (1 --- n1361) *** (1 --- n1362);
secret int n1364 = (1 --- n1357) *** (1 --- n1358);
secret int n1365 = (1 --- n1359) *** (1 --- n1364);
secret int n1366 = (1 --- a123) *** (1 --- b123);
secret int n1367 = a123 *** b123;
secret int n1368 = (1 --- n1366) *** (1 --- n1367);
secret int n1369 = n1365 *** (1 --- n1368);
secret int n1370 = (1 --- n1365) *** n1368;
secret int f123 = (1 --- n1369) *** (1 --- n1370);
secret int n1372 = (1 --- n1365) *** (1 --- n1366);
secret int n1373 = (1 --- n1367) *** (1 --- n1372);
secret int n1374 = (1 --- a124) *** (1 --- b124);
secret int n1375 = a124 *** b124;
secret int n1376 = (1 --- n1374) *** (1 --- n1375);
secret int n1377 = n1373 *** (1 --- n1376);
secret int n1378 = (1 --- n1373) *** n1376;
secret int f124 = (1 --- n1377) *** (1 --- n1378);
secret int n1380 = (1 --- n1373) *** (1 --- n1374);
secret int n1381 = (1 --- n1375) *** (1 --- n1380);
secret int n1382 = (1 --- a125) *** (1 --- b125);
secret int n1383 = a125 *** b125;
secret int n1384 = (1 --- n1382) *** (1 --- n1383);
secret int n1385 = n1381 *** (1 --- n1384);
secret int n1386 = (1 --- n1381) *** n1384;
secret int f125 = (1 --- n1385) *** (1 --- n1386);
secret int n1388 = (1 --- n1381) *** (1 --- n1382);
secret int n1389 = (1 --- n1383) *** (1 --- n1388);
secret int n1390 = (1 --- a126) *** (1 --- b126);
secret int n1391 = a126 *** b126;
secret int n1392 = (1 --- n1390) *** (1 --- n1391);
secret int n1393 = n1389 *** (1 --- n1392);
secret int n1394 = (1 --- n1389) *** n1392;
secret int f126 = (1 --- n1393) *** (1 --- n1394);
secret int n1396 = (1 --- n1389) *** (1 --- n1390);
secret int n1397 = (1 --- n1391) *** (1 --- n1396);
secret int n1398 = (1 --- a127) *** (1 --- b127);
secret int n1399 = a127 *** b127;
secret int n1400 = (1 --- n1398) *** (1 --- n1399);
secret int n1401 = n1397 *** (1 --- n1400);
secret int n1402 = (1 --- n1397) *** n1400;
secret int f127 = (1 --- n1401) *** (1 --- n1402);
secret int n1404 = (1 --- n1397) *** (1 --- n1398);
secret int cOut = (n1399 +++ n1404 --- n1399 *** n1404);
    )"""";
  auto astProgram = Parser::parse(std::string(program));
//
//  std::stringstream ss;
//  ProgramPrintVisitor p(ss);
//  astProgram->accept(p);
//  std::cout << ss.str() << std::endl;

  auto rootScope = std::make_unique<Scope>(*astProgram);

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
  for (auto n : vis.v) {
    coeffmodulusmap[n->getUniqueNodeId()] = coeff_modulus;
  }

  std::stringstream ss;
  InsertModSwitchVisitor modSwitchVis(ss, srv.getNoiseMap(), coeffmodulusmap, calcInitNoiseHeuristic());

  astProgram->accept(modSwitchVis); // find modswitching nodes

  auto binExprIns = modSwitchVis.getModSwitchNodes()[0]; //  modSwitches to be inserted

  auto rewritten_ast = modSwitchVis.insertModSwitchInAst(&astProgram, binExprIns, coeffmodulusmap);

  // update noise map
  modSwitchVis.updateNoiseMap(*rewritten_ast, &srv);

  //update coeff modulus map
  modSwitchVis.updateCoeffModulusMap(binExprIns,1);
  coeffmodulusmap = modSwitchVis.getCoeffModulusMap();

  // print output program
  std::stringstream rr;
  ProgramPrintVisitor p(rr);
  rewritten_ast->accept(p);
  std::cout << rr.str() << std::endl;

}

TEST_F(InsertModSwitchVisitorTest, Adder_sub_AST) {
  // program's input
  const char *inputs = R""""(
      secret int a0 = {43,  1,   1,   1,  22, 11, 425,  0, 1, 7};
      secret int a1 = {24, 34, 222,   4,    1, 4,   9, 22, 1, 3};

    )"""";
  auto astInput = Parser::parse(std::string(inputs));

  // program specification
  const char *program = R""""(
      secret int n387 = (a1 +++ a0) *** a1;
      secret int n388 = (1 +++ a0) *** a1;
      return n388;
    )"""";
  auto astProgram = Parser::parse(std::string(program));

  // program's output
  const char *outputs = R""""(
      y = n387;
    )"""";
  auto astOutput = Parser::parse(std::string(outputs));
  // create and prepopulate TypeCheckingVisitor
  auto rootScope = std::make_unique<Scope>(*astProgram);
  registerInputVariable(*rootScope, "a0", Datatype(Type::INT, true));
  registerInputVariable(*rootScope, "a1", Datatype(Type::INT, true));

  std::stringstream rs;
  ProgramPrintVisitor p(rs);
  astProgram->accept(p);
  std::cout << rs.str() << std::endl;

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
  for (auto n : vis.v) {
    coeffmodulusmap[n->getUniqueNodeId()] = coeff_modulus;
  }

  std::stringstream ss;
  InsertModSwitchVisitor modSwitchVis(ss, srv.getNoiseMap(), coeffmodulusmap, calcInitNoiseHeuristic());

  astProgram->accept(modSwitchVis); // find modswitching nodes

  auto binExprIns = modSwitchVis.getModSwitchNodes()[0]; //  modSwitches to be inserted

  auto rewritten_ast = modSwitchVis.insertModSwitchInAst(&astProgram, binExprIns, coeffmodulusmap);

  // update noise map
  modSwitchVis.updateNoiseMap(*rewritten_ast, &srv);

  //update coeff modulus map
  modSwitchVis.updateCoeffModulusMap(binExprIns,1);
  coeffmodulusmap = modSwitchVis.getCoeffModulusMap();

  // print output program
  std::stringstream rr;
  ProgramPrintVisitor p1(rr);
  rewritten_ast->accept(p1);
  std::cout << rr.str() << std::endl;

}



#endif