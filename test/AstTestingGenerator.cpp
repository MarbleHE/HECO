#include "AstTestingGenerator.h"
#include <Operator.h>
#include "BinaryExpr.h"
#include "UnaryExpr.h"
#include "VarAssignm.h"
#include "Return.h"
#include "Function.h"
#include "If.h"
#include "LogicalExpr.h"
#include "CallExternal.h"
#include "Block.h"
#include "While.h"
#include "Call.h"

void AstTestingGenerator::generateAst(int id, Ast &ast) {
  // generate AST
  std::map<int, std::function<void(Ast &)> > call = {
      {1, genAstRewritingOne},
      {2, genAstRewritingTwo},
      {3, genAstRewritingThree},
      {4, genAstRewritingFour},
      {5, genAstRewritingFive},
      {6, genAstRewritingSix},
      {7, genAstEvalOne},
      {8, genAstEvalTwo},
      {9, genAstEvalThree},
      {10, genAstEvalFour},
      {11, genAstEvalFive},
      {12, genAstEvalSix},
      {13, genAstEvalSeven}
  };
  auto it = call.find(id);
  if (it == call.end()) throw std::logic_error("Cannot continue. Invalid id given!");
  it->second(ast);
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

void AstTestingGenerator::genAstEvalOne(Ast &ast) {
  // int computePrivate(int inputA, int inputB, int inputC)
  auto func = new Function("computeAvg");
  auto funcParams = new std::vector<FunctionParameter>();
  funcParams->emplace_back("int", new Variable("width"));
  funcParams->emplace_back("int", new Variable("length"));
  funcParams->emplace_back("int", new Variable("depth"));
  func->setParams(funcParams);

  // int prod = inputA * inputB * inputC
  func->addStatement(new VarDecl("prod", "int",
                                 new BinaryExpr(
                                     new Variable("width"),
                                     OpSymb::multiplication,
                                     new BinaryExpr(
                                         new Variable("length"),
                                         OpSymb::multiplication,
                                         new Variable("depth")))));

  // return prod / 3;
  func->addStatement(new Return(new BinaryExpr(
      new Variable("prod"),
      OpSymb::division,
      new LiteralInt(3))));

  ast.setRootNode(func);
}

void AstTestingGenerator::genAstEvalTwo(Ast &ast) {
  // int computePrivate(int inputA, int inputB, int inputC)
  auto func = new Function("computePrivate");
  auto funcParams = new std::vector<FunctionParameter>();
  funcParams->emplace_back("int", new Variable("inputA"));
  funcParams->emplace_back("int", new Variable("inputB"));
  funcParams->emplace_back("bool", new Variable("takeIf"));
  func->setParams(funcParams);

  // int prod = inputA * inputB;
  func->addStatement(new VarDecl("prod", "int",
                                 new BinaryExpr(
                                     new Variable("inputA"),
                                     OpSymb::multiplication,
                                     new Variable("inputB"))));


  // if (takeIf) { prod = prod % 12; } else { prod = prod - 21; }
  func->addStatement(new If(
      new LogicalExpr(new Variable("takeIf"), OpSymb::equal, new LiteralBool(true)),
      new VarAssignm("prod",
                     new BinaryExpr(
                         new Variable("prod"),
                         OpSymb::modulo,
                         new LiteralInt(12))),
      new VarAssignm("prod",
                     new BinaryExpr(
                         new Variable("prod"),
                         OpSymb::subtraction,
                         new LiteralInt(21)))));

  // return prod;
  func->addStatement(new Return(new Variable("prod")));

  ast.setRootNode(func);
}

void AstTestingGenerator::genAstEvalThree(Ast &ast) {
  // bool computeLogical(int inputA, bool strong, bool negate, int inputB)
  auto func = new Function("computePrivate");
  auto funcParams = new std::vector<FunctionParameter>();
  funcParams->emplace_back("int", new Variable("inputA"));
  funcParams->emplace_back("bool", new Variable("strong"));
  funcParams->emplace_back("bool", new Variable("negate"));
  funcParams->emplace_back("bool", new Variable("inputB"));
  func->setParams(funcParams);

  // if (strong == true) { inputA = inputA * 42; } else { inputA = inputA + 42; }
  func->addStatement(new If(
      new LogicalExpr(new Variable("strong"), OpSymb::equal, new LiteralBool(true)),
      new Block(new VarAssignm("inputA",
                               new BinaryExpr(
                                   new Variable("inputA"),
                                   OpSymb::multiplication,
                                   new LiteralInt(42)))),
      new Block(new VarAssignm("inputA",
                               new BinaryExpr(
                                   new Variable("inputA"),
                                   OpSymb::addition,
                                   new LiteralInt(42))))));

  // if (negate) { inputA = -inputA }
  func->addStatement(new If(
      new LogicalExpr(new Variable("negate"), OpSymb::equal, new LiteralBool(true)),
      new VarAssignm("inputA", new UnaryExpr(OpSymb::negation, new Variable("inputA")))));

  // return inputA >= inputB
  func->addStatement(
      new Return(
          new LogicalExpr(new Variable("inputA"), OpSymb::greaterEqual, new Variable("inputB"))));

  ast.setRootNode(func);
}

void AstTestingGenerator::genAstEvalFour(Ast &ast) {
  auto func = new Function("concatString");
  auto funcParams = new std::vector<FunctionParameter>();
  funcParams->emplace_back("string", new Variable("strA"));
  funcParams->emplace_back("string", new Variable("strB"));
  func->setParams(funcParams);

  // return strA + strB
  func->addStatement(new Return(new BinaryExpr(new Variable("strA"), OpSymb::addition, new Variable("strB"))));

  ast.setRootNode(func);
}

void AstTestingGenerator::genAstEvalFive(Ast &ast) {
  // int determineSuitableX(int encryptedA, int encryptedB) {...}
  Function* func = dynamic_cast<Function*>(ast.setRootNode(new Function("determineSuitableX")));
  func->addParameter(new FunctionParameter("int", new Variable("encryptedA")));
  func->addParameter(new FunctionParameter("int", new Variable("encryptedB")));
  func->addParameter(new FunctionParameter("int", new Variable("randInt")));

  // bool b = encryptedA < 2;
  func->addStatement(
      new VarDecl("b", "bool",
                  new LogicalExpr(
                      new Variable("encryptedA"),
                      OpSymb::LogCompOp::smaller,
                      new LiteralInt(2))));

  // int sum = 0;
  func->addStatement(new VarDecl("sum", "int", 0));

  // while (randInt > 0 && !b == true) { sum = sum + encryptedB; randInt--; };
  func->addStatement(new While(
      // condition: (randInt > 0 && !b == true)
      new LogicalExpr(
          // randInt > 0
          new LogicalExpr(
              new Variable("randInt"),
              OpSymb::LogCompOp::greater,
              new LiteralInt(0)),
          OpSymb::LogCompOp::logicalAnd,
          // !b == true
          new LogicalExpr(
              new UnaryExpr(OpSymb::UnaryOp::negation, new Variable("b")),
              OpSymb::LogCompOp::equal,
              new LiteralBool(true))),
      // body: { sum = sum + encryptedB; randInt--; };
      new Block(
          new std::vector<AbstractStatement*>{
              // sum = sum + encryptedB
              new VarAssignm("sum", new BinaryExpr(
                  new Variable("sum"),
                  OpSymb::BinaryOp::addition,
                  new Variable("encryptedB"))),
              // randInt = randInt-1;
              new VarAssignm("randInt", new BinaryExpr(
                  new Variable("randInt"),
                  OpSymb::BinaryOp::subtraction,
                  new LiteralInt(1)))})));

  // return sum;
  func->addStatement(
      new Return(new Variable("sum")));
}

void AstTestingGenerator::genAstEvalSix(Ast &ast) {
  // int nestedCall() { ...
  auto* fnc = new Function("nestedCall");
  ast.setRootNode(fnc);

  // int result = computeSecret(33);
  // -> computeSecret(int inputA) { return inputA * 32 }
  fnc->addStatement(new VarAssignm("result", new Call(
      {new FunctionParameter("int", new LiteralInt(33))},
      new Function("computeSecret",
                   {FunctionParameter("int", new Variable("inputA"))},
                   {new Return(
                       new BinaryExpr(
                           new Variable("inputA"),
                           OpSymb::multiplication,
                           new LiteralInt(32)))
                   }))));

  // return result;
  fnc->addStatement(new Return(new Variable("result")));
}

void AstTestingGenerator::genAstEvalSeven(Ast &ast) {
  // int nestedCall() { ...
  auto* fnc = new Function("nestedCall");
  ast.setRootNode(fnc);

  // int result = computeSecret(33);
  // -> computeSecret(int inputA) { return inputA * 32 }
  fnc->addStatement(new VarAssignm("result", new Call(
      {new FunctionParameter("int", new BinaryExpr(new LiteralInt(11), OpSymb::addition, new LiteralInt(213)))},
      new Function("computeSecret",
                   {FunctionParameter("int", new Variable("inputA"))},
                   {new Return(
                       new BinaryExpr(
                           new Variable("inputA"),
                           OpSymb::multiplication,
                           new LiteralInt(32)))
                   }))));

  // return result;
  fnc->addStatement(new Return(new Variable("result")));
}
