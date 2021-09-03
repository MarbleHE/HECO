#include <ast_opt/visitor/TypeCheckingVisitor.h>
#include <ast_opt/runtime/RuntimeVisitor.h>
#include <ast_opt/parser/Parser.h>
#include <ast_opt/runtime/DummyCiphertextFactory.h>
#include "ast_opt/compiler/Compiler.h"

OutputIdentifierValuePairs Compiler::compileAst(std::unique_ptr<AbstractNode> astProgram,
                                                std::unique_ptr<AbstractNode> astInput,
                                                std::unique_ptr<AbstractNode> astOutput) {

  auto scf = std::make_unique<DummyCiphertextFactory>();
  auto tcv = std::make_unique<TypeCheckingVisitor>();

  // create and prepopulate TypeCheckingVisitor
  auto registerInputVariable = [&tcv](Scope &rootScope, const std::string &identifier, Datatype datatype) {
    auto scopedIdentifier = std::make_unique<ScopedIdentifier>(rootScope, identifier);
    rootScope.addIdentifier(identifier);
    tcv->addVariableDatatype(*scopedIdentifier, datatype);
  };

  auto rootScope = std::make_unique<Scope>(*astProgram);

  // TODO [mh]: parse identifiers and types from astInput?
//  registerInputVariable(*rootScope, "x", Datatype(Type::INT, false));
//  registerInputVariable(*rootScope, "y", Datatype(Type::INT, false));
//  registerInputVariable(*rootScope, "size", Datatype(Type::INT, false));

  tcv->setRootScope(std::move(rootScope));
  astProgram->accept(*tcv);

  // run the program and get its output
  //TODO: Change it so that by passing in an empty secretTaintingMap, we can get the RuntimeVisitor to execute everything "in the clear"!
  auto empty = std::unordered_map<std::string, bool>();
  RuntimeVisitor srv(*scf, *astInput, empty);
  srv.executeAst(*astProgram);
  return srv.getOutput(*astOutput);
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

  return Compiler::compileAst(std::move(programAst), std::move(inputAst), std::move(outputAst));
}

OutputIdentifierValuePairs Compiler::compileJson(std::string program,
                                                 std::string input,
                                                 std::vector<std::string> outputIdentifiers) {
  auto programAst = Parser::parseJson(program);
  auto inputAst = Parser::parseJson(input);
  auto outputAst = Compiler::buildOutputBlock(outputIdentifiers);

  return Compiler::compileAst(std::move(programAst), std::move(inputAst), std::move(outputAst));
}
