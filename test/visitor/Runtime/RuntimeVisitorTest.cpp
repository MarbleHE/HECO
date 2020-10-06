#include "include/ast_opt/ast/AbstractNode.h"
#include "include/ast_opt/parser/Parser.h"
#include "include/ast_opt/ast/Literal.h"
#include "include/ast_opt/visitor/Runtime/RuntimeVisitor.h"
#include "include/ast_opt/visitor/Runtime/SealCiphertextFactory.h"

#include "gtest/gtest.h"

class RuntimeVisitorTest : public ::testing::Test {
 protected:
  std::unique_ptr<SealCiphertextFactory> scf;

  void SetUp() override {
    scf = std::make_unique<SealCiphertextFactory>(4096);
  }
};

TEST_F(RuntimeVisitorTest, programExecution) { /* NOLINT */
  const char *inputChars = R""""(
    public int main(secret int N) {
      int sum = 2442;
      if (N < 5) {
        sum = sum-N;
      }
      return sum;
    }
    )"""";
  std::vector<std::reference_wrapper<AbstractNode>> createdNodes;
  auto inputAst = Parser::parse(std::string(inputChars), createdNodes);

}

TEST_F(RuntimeVisitorTest, test) { /* NOLINT */
  // program's input
  const char *inputs = R""""(
      secret int __input0__ = {43, 1, 1, 1, 22, 11, 425, 0, 1, 0};
      int __input1__ = {43, 1, 1, 1, 22, 11, 425, 0, 1, 0};
    )"""";
  auto astInput = Parser::parse(std::string(inputs));

  // program specification
  const char *program = R""""(
      __input0__ = rotate(__input0__, 4);
    )"""";
  auto astProgram = Parser::parse(std::string(program));

  // program's output
  const char *outputs = R""""(
      y = __input0__;
    )"""";
  auto astOutput = Parser::parse(std::string(outputs));


  // create a SpecialRuntimeVisitor instance
  SpecialRuntimeVisitor srv(*scf, *astInput);

//  // add identifiers to (outermost) root scope, otherwise resolveIdentifier will fail
//  srv.setRootScope(std::make_unique<Scope>(*astProgram));
//  srv.getRootScope().addIdentifier("__input0__");
//  srv.getRootScope().addIdentifier("__input1__");

  // run the program
  astProgram->accept(srv);



  // print the output
  srv.printOutput(*astOutput);
}

