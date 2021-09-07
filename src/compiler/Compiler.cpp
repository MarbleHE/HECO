#include <ast_opt/visitor/TypeCheckingVisitor.h>
#include <ast_opt/runtime/RuntimeVisitor.h>
#include <ast_opt/parser/Parser.h>
#include <ast_opt/runtime/DummyCiphertextFactory.h>
#include <ast_opt/parser/Errors.h>
#include <ast_opt/compiler/Compiler.h>
#include <ast_opt/utilities/NodeUtils.h>

OutputIdentifierValuePairs Compiler::compileAst(std::unique_ptr<AbstractNode> programAst,
                                                std::unique_ptr<Block> inputBlock,
                                                std::unique_ptr<AbstractNode> outputAst) {

  auto scf = std::make_unique<DummyCiphertextFactory>();
  auto tcv = std::make_unique<TypeCheckingVisitor>();

  // create and prepopulate TypeCheckingVisitor
  auto registerInputVariable = [&tcv](Scope &rootScope, const std::string &identifier, Datatype datatype) {
    auto scopedIdentifier = std::make_unique<ScopedIdentifier>(rootScope, identifier);
    rootScope.addIdentifier(identifier);
    tcv->addVariableDatatype(*scopedIdentifier, datatype);
  };

  // Parse the input block and add detected variables to the root scope.
  auto rootScope = std::make_unique<Scope>(*programAst);

  for (auto stmt : inputBlock->getStatements()) {
    if (auto variable_declaration = dynamic_cast<VariableDeclaration *>(&stmt.get())) {
      if (auto variable = dynamic_cast<Variable *>(&variable_declaration->getTarget())) {
        registerInputVariable(*rootScope,
                              variable->getIdentifier(),
                              variable_declaration->getDatatype());
      }
      else {
        stork::runtime_error("Invalid input block: all statements must contain a variable name on the LHS.");
      }
    }
    else {
      stork::runtime_error("Invalid input block: must only consists of variable declarations.");
    }
  }

  tcv->setRootScope(std::move(rootScope));
  programAst->accept(*tcv);

  // run the program and get its output
  //TODO: Change it so that by passing in an empty secretTaintingMap, we can get the RuntimeVisitor to execute everything "in the clear"!
  auto empty = std::unordered_map<std::string, bool>();
  RuntimeVisitor srv(*scf, *inputBlock, empty);
  srv.executeAst(*programAst);
  return srv.getOutput(*outputAst);
}

std::unique_ptr<AbstractNode> Compiler::buildOutputBlock(std::vector<std::string> outputIdentifiers) {
  std::vector<std::unique_ptr<AbstractStatement>> statements;

  for (auto identifier : outputIdentifiers) {
    auto assignment = std::make_unique<Assignment>(std::make_unique<Variable>(identifier), std::make_unique<Variable>(identifier));
    statements.emplace_back(std::move(assignment));
  }
  return std::make_unique<Block>(std::move(statements));
}

OutputIdentifierValuePairs Compiler::compile(std::string program,
                                             std::string input,
                                             std::vector<std::string> outputIdentifiers) {

  auto programAst = Parser::parse(program);
  auto inputAst = Parser::parse(input);
  auto outputAst = Compiler::buildOutputBlock(outputIdentifiers);

  // The input must be a block by convention, otherwise, throw an error
  auto inputBlock = castUniquePtr<AbstractNode, Block>(std::move(inputAst));

  return Compiler::compileAst(std::move(programAst), std::move(inputBlock), std::move(outputAst));
}

OutputIdentifierValuePairs Compiler::compileJson(std::string program,
                                                 std::string input,
                                                 std::vector<std::string> outputIdentifiers) {
  return Compiler::compileJson(Parser::parseJson(program), input, outputIdentifiers);
}

OutputIdentifierValuePairs Compiler::compileJson(std::unique_ptr<AbstractNode> programAst,
                                                 std::string input,
                                                 std::vector<std::string> outputIdentifiers) {
  auto inputAst = Parser::parseJson(input);
  auto outputAst = Compiler::buildOutputBlock(outputIdentifiers);

  // The input must be a block by convention, otherwise, throw an error
  auto inputBlock = castUniquePtr<AbstractNode, Block>(std::move(inputAst));

  return Compiler::compileAst(std::move(programAst), std::move(inputBlock), std::move(outputAst));
}
