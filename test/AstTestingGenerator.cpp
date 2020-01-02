#include "AstTestingGenerator.h"
#include <Operator.h>
#include "BinaryExpr.h"
#include "VarAssignm.h"
#include "Return.h"
#include "Function.h"
#include "If.h"
#include "LogicalExpr.h"
#include "CallExternal.h"

void AstTestingGenerator::generateAst(int id, Ast &ast) {
  switch (id) {
    case 1: genAstRewritingOne(ast);
      break;
    case 2: genAstRewritingTwo(ast);
      break;
    case 3: genAstRewritingThree(ast);
      break;
    case 4: genAstRewritingFour(ast);
      break;
    case 5: genAstRewritingFive(ast);
      break;
    case 6: genAstRewritingSix(ast);
      break;
    default: throw std::logic_error("Cannot continue. Invalid id given!");
  }
}

void AstTestingGenerator::genAstRewritingOne(Ast &ast) {
  // int computePrivate(int inputA, int inputB, int inputC)
  auto func = new Function("computePrivate");
  auto funcParams = new std::vector<FunctionParameter>();
  funcParams->emplace_back("int", new Variable("inputA"));
  funcParams->emplace_back("int", new Variable("inputB"));
  funcParams->emplace_back("int", new Variable("inputC"));
  func->setParams(funcParams);

  // int prod = [inputA * [inputB * inputC]]
  func->addStatement(new VarDecl("prod", "int",
                                 new BinaryExpr(
                                     new Variable("inputA"),
                                     OpSymb::multiplication,
                                     new BinaryExpr(
                                         new Variable("inputB"),
                                         OpSymb::multiplication,
                                         new Variable("inputC")))));

  // return prod / 3;
  func->addStatement(new Return(new BinaryExpr(
      new Variable("prod"),
      OpSymb::division,
      new LiteralInt(3))));

  ast.setRootNode(func);
}

void AstTestingGenerator::genAstRewritingTwo(Ast &ast) {
  // int computePrivate(int inputA, int inputB, int inputC)
  auto func = new Function("computePrivate");
  auto funcParams = new std::vector<FunctionParameter>();
  funcParams->emplace_back("int", new Variable("inputA"));
  funcParams->emplace_back("int", new Variable("inputB"));
  funcParams->emplace_back("int", new Variable("inputC"));
  func->setParams(funcParams);

  // int prod = inputA * inputB;
  func->addStatement(new VarDecl("prod", "int",
                                 new BinaryExpr(
                                     new Variable("inputA"),
                                     OpSymb::multiplication,
                                     new Variable("inputB"))));

  // prod = prod * inputC
  func->addStatement(
      new VarAssignm("prod", new BinaryExpr(
          new Variable("prod"),
          OpSymb::multiplication,
          new Variable("inputC"))));

  // return prod / 3;
  func->addStatement(new Return(new BinaryExpr(
      new Variable("prod"),
      OpSymb::division,
      new LiteralInt(3))));

  ast.setRootNode(func);
}

void AstTestingGenerator::genAstRewritingThree(Ast &ast) {
  // int computePrivate(int inputA, int inputB, int inputC)
  auto func = new Function("computePrivate");
  auto funcParams = new std::vector<FunctionParameter>();
  funcParams->emplace_back("int", new Variable("inputA"));
  funcParams->emplace_back("int", new Variable("inputB"));
  funcParams->emplace_back("int", new Variable("inputC"));
  func->setParams(funcParams);

  // int prod = inputA * inputB;
  func->addStatement(new VarDecl("prod", "int",
                                 new BinaryExpr(
                                     new Variable("inputA"),
                                     OpSymb::multiplication,
                                     new Variable("inputB"))));

  // int rInt = rand()
  func->addStatement(new VarDecl("rInt", "int", new CallExternal("std::rand")));

  // prod = prod * inputC
  func->addStatement(
      new VarAssignm("prod", new BinaryExpr(
          new Variable("prod"),
          OpSymb::multiplication,
          new Variable("inputC"))));

  // return prod / 3;
  func->addStatement(new Return(new BinaryExpr(
      new Variable("prod"),
      OpSymb::division,
      new LiteralInt(3))));

  ast.setRootNode(func);
}

void AstTestingGenerator::genAstRewritingFour(Ast &ast) {
  // int computePrivate(int inputA, int inputB, int inputC)
  auto func = new Function("computePrivate");
  auto funcParams = new std::vector<FunctionParameter>();
  funcParams->emplace_back("int", new Variable("inputA"));
  funcParams->emplace_back("int", new Variable("inputB"));
  funcParams->emplace_back("int", new Variable("inputC"));
  func->setParams(funcParams);

  // int prod = inputA * inputB;
  func->addStatement(new VarDecl("prod", "int",
                                 new BinaryExpr(
                                     new Variable("inputA"),
                                     OpSymb::multiplication,
                                     new Variable("inputB"))));


  // if (prod > 42) { prod = prod * inputC; }
  func->addStatement(new If(
      new LogicalExpr(new Variable("prod"), OpSymb::greater, new LiteralInt(42)),
      new VarAssignm("prod", new BinaryExpr(
          new Variable("prod"),
          OpSymb::multiplication,
          new Variable("inputC")))));

  // return prod / 3;
  func->addStatement(new Return(new BinaryExpr(
      new Variable("prod"),
      OpSymb::division,
      new LiteralInt(3))));

  ast.setRootNode(func);
}

void AstTestingGenerator::genAstRewritingFive(Ast &ast) {
  // int computePrivate(int inputA, int inputB, int inputC)
  auto func = new Function("computePrivate");
  auto funcParams = new std::vector<FunctionParameter>();
  funcParams->emplace_back("int", new Variable("inputA"));
  funcParams->emplace_back("int", new Variable("inputB"));
  funcParams->emplace_back("int", new Variable("inputC"));
  func->setParams(funcParams);

  // int prod = inputA * inputB;
  func->addStatement(new VarDecl("prod", "int",
                                 new BinaryExpr(
                                     new Variable("inputA"),
                                     OpSymb::multiplication,
                                     new Variable("inputB"))));

  // prod = prod * inputC
  func->addStatement(
      new VarAssignm("argPow", new BinaryExpr(
          new Variable("inputC"),
          OpSymb::multiplication,
          new Variable("inputC"))));

  // return prod / 3;
  func->addStatement(new Return(new BinaryExpr(
      new Variable("prod"),
      OpSymb::division,
      new LiteralInt(3))));

  ast.setRootNode(func);
}

void AstTestingGenerator::genAstRewritingSix(Ast &ast) {
  // int computePrivate(int inputA, int inputB, int inputC)
  auto func = new Function("computePrivate");
  auto funcParams = new std::vector<FunctionParameter>();
  funcParams->emplace_back("int", new Variable("inputA"));
  funcParams->emplace_back("int", new Variable("inputB"));
  funcParams->emplace_back("int", new Variable("inputC"));
  func->setParams(funcParams);

  // int prod = inputA * inputB;
  func->addStatement(new VarDecl("prod", "int",
                                 new BinaryExpr(
                                     new Variable("inputA"),
                                     OpSymb::multiplication,
                                     new Variable("inputB"))));

  // int prod = inputA * inputB;
  func->addStatement(new VarDecl("prod2", "int",
                                 new BinaryExpr(
                                     new Variable("prod"),
                                     OpSymb::multiplication,
                                     new Variable("inputC"))));

  // return prod / 3;
  func->addStatement(new Return(new BinaryExpr(
      new Variable("prod"),
      OpSymb::division,
      new LiteralInt(3))));

  ast.setRootNode(func);
}
