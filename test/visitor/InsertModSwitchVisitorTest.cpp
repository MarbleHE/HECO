#include <gmp.h>
#include "include/ast_opt/parser/Parser.h"
#include "include/ast_opt/runtime/RuntimeVisitor.h"
#include "include/ast_opt/runtime/SimulatorCiphertextFactory.h"
#include "ast_opt/visitor/InsertModSwitchVisitor.h"
#include <ast_opt/visitor/GetAllNodesVisitor.h>
#include "../ASTComparison.h"
#include "ast_opt/visitor/ProgramPrintVisitor.h"
#include "ast_opt/visitor/PrintVisitor.h"
#include "ast_opt/utilities/PerformanceSeal.h"

#include "gtest/gtest.h"
#ifdef HAVE_SEAL_BFV

class InsertModSwitchVisitorTest : public ::testing::Test {
 protected:
  std::unique_ptr<SimulatorCiphertextFactory> scf;
  std::unique_ptr<TypeCheckingVisitor> tcv;

  void SetUp() override {
    scf = std::make_unique<SimulatorCiphertextFactory>(8192);
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

  // print coeff modulus map
  for (auto &n : vis.v) {
   std::cout << n->toString(false) << " " << coeffmodulusmap[n->getUniqueNodeId()].size() << std::endl;
  }


  // print output program
  std::stringstream rr;
  ProgramPrintVisitor p(rr);
  rewritten_ast->accept(p);
  std::cout << rr.str() << std::endl;


  //In this case, asts should be identical
  ASSERT_NE(rewritten_ast, nullptr);
//  compareAST(*astProgram_expected, *rewritten_ast);

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

//  for (int j = 0; j < coeff_modulus.size(); j++) {
//    std::cout << coeff_modulus[j].bit_count() << " ";
//  }

  // remove the last prime for binaryOp.getLeft() in coeffmodulus map (our goal is to have two modswitches inserted...)
  coeffmodulusmap["Variable_33"].pop_back();

  std::cout << "Initial Noise Heur: " << calcInitNoiseHeuristic() << std::endl;

  auto tamperedNoiseMap = srv.getNoiseMap();
  tamperedNoiseMap["Variable_33"] = 32;
  tamperedNoiseMap["Variable_35"] = 32;


// for (auto n : vis.v) {
//   std::cout << "Type: " << n->toString(false) << " ID: " << n->getUniqueNodeId() << std::endl;
// }

  std::stringstream rr;
  InsertModSwitchVisitor modSwitchVis(rr, tamperedNoiseMap, coeffmodulusmap, calcInitNoiseHeuristic());
  astProgram->accept(modSwitchVis); // find modswitching nodes

  std::cout << modSwitchVis.getModSwitchNodes().size() << " potential modSwitch insertion site(s) found:" << std::endl;

  auto binExprIns = modSwitchVis.getModSwitchNodes()[0]; //  modSwitches to be inserted

  auto rewritten_ast = modSwitchVis.insertModSwitchInAst(&astProgram, binExprIns, coeffmodulusmap);

  std::stringstream rs;
  ProgramPrintVisitor q(rs);
  rewritten_ast->accept(q);
  std::cout << rs.str() << std::endl;

}



#endif