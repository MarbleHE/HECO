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
#include "Group.h"

static std::map<int, std::function<void(Ast &)> > call = {
    {1, AstTestingGenerator::_genAstRewritingOne},
    {2, AstTestingGenerator::_genAstRewritingTwo},
    {3, AstTestingGenerator::_genAstRewritingThree},
    {4, AstTestingGenerator::_genAstRewritingFour},
    {5, AstTestingGenerator::_genAstRewritingFive},
    {6, AstTestingGenerator::_genAstRewritingSix},
    {7, AstTestingGenerator::_genAstEvalOne},
    {8, AstTestingGenerator::_genAstEvalTwo},
    {9, AstTestingGenerator::_genAstEvalThree},
    {10, AstTestingGenerator::_genAstEvalFour},
    {11, AstTestingGenerator::_genAstEvalFive},
    {12, AstTestingGenerator::_genAstEvalSix},
    {13, AstTestingGenerator::_genAstEvalSeven},
    {14, AstTestingGenerator::_genAstPrintVisitorOne},
    {15, AstTestingGenerator::_genAstPrintVisitorTwo}
};

void AstTestingGenerator::generateAst(int id, Ast &ast) {
  auto it = call.find(id);
  if (it == call.end()) throw std::logic_error("Cannot continue. Invalid id given!");
  it->second(ast);
}

int AstTestingGenerator::getLargestId() {
  return call.size();
}

void AstTestingGenerator::_genAstRewritingOne(Ast &ast) {
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

void AstTestingGenerator::_genAstRewritingTwo(Ast &ast) {
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

void AstTestingGenerator::_genAstRewritingThree(Ast &ast) {
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

void AstTestingGenerator::_genAstRewritingFour(Ast &ast) {
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

void AstTestingGenerator::_genAstRewritingFive(Ast &ast) {
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

void AstTestingGenerator::_genAstRewritingSix(Ast &ast) {
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

void AstTestingGenerator::_genAstEvalOne(Ast &ast) {
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

void AstTestingGenerator::_genAstEvalTwo(Ast &ast) {
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

void AstTestingGenerator::_genAstEvalThree(Ast &ast) {
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

void AstTestingGenerator::_genAstEvalFour(Ast &ast) {
  auto func = new Function("concatString");
  auto funcParams = new std::vector<FunctionParameter>();
  funcParams->emplace_back("string", new Variable("strA"));
  funcParams->emplace_back("string", new Variable("strB"));
  func->setParams(funcParams);

  // return strA + strB
  func->addStatement(new Return(new BinaryExpr(new Variable("strA"), OpSymb::addition, new Variable("strB"))));

  ast.setRootNode(func);
}

void AstTestingGenerator::_genAstEvalFive(Ast &ast) {
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
  func->addStatement(new VarDecl("sum", 0));

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

void AstTestingGenerator::_genAstEvalSix(Ast &ast) {
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

void AstTestingGenerator::_genAstEvalSeven(Ast &ast) {
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

void AstTestingGenerator::_genAstPrintVisitorOne(Ast &ast) {
  // int computePrivate(int x) { ... }
  Function* func = dynamic_cast<Function*>(ast.setRootNode(new Function("computePrivate")));
  func->addParameter(new FunctionParameter("int", new Variable("x")));

  // int a = 4;
  func->addStatement(new VarDecl("a", 4));

  // int k;
  func->addStatement(new VarDecl("k", "int", nullptr));

  // if ( x > 32 ) { k = x * a; } else { k = (x * a) + 42; }
  func->addStatement(new If(
      // if (x > 32)
      new LogicalExpr(new Variable("x"), OpSymb::LogCompOp::greater, 32),
      // { k = x * a}
      new Block(
          new VarAssignm("k",
                         new BinaryExpr(
                             new Variable("x"),
                             OpSymb::BinaryOp::multiplication,
                             new Variable("a")))),
      // else { k = (x * a) + 42; }
      new Block(
          new VarAssignm("k",
                         new BinaryExpr(
                             new Group(
                                 new BinaryExpr(
                                     new Variable("x"),
                                     OpSymb::BinaryOp::multiplication,
                                     new Variable("a"))),
                             OpSymb::BinaryOp::addition,
                             42)))));

  // return k
  func->addStatement(new Return(new Variable("k")));
}

void AstTestingGenerator::_genAstPrintVisitorTwo(Ast &ast) {
// int determineSuitableX(int encryptedA, int encryptedB) {...}
  Function* func = dynamic_cast<Function*>(ast.setRootNode(new Function("determineSuitableX")));
  func->addParameter(new FunctionParameter("int", new Variable("encryptedA")));
  func->addParameter(new FunctionParameter("int", new Variable("encryptedB")));

  // int randInt = rand() % 42;
  func->addStatement(
      new VarDecl("randInt", "int",
                  new BinaryExpr(
                      new CallExternal("std::rand"),
                      OpSymb::BinaryOp::modulo,
                      new LiteralInt(42))));

  // bool b = encryptedA < 2;
  func->addStatement(
      new VarDecl("b", "bool",
                  new LogicalExpr(
                      new Variable("encryptedA"),
                      OpSymb::LogCompOp::smaller,
                      new LiteralInt(2))));

  // int sum = 0;
  func->addStatement(new VarDecl("sum", 0));

  // while (randInt > 0 && !b == true) { sum = sum + encryptedB; randInt--; };
  func->addStatement(new While(
      // (randInt > 0 && !b == true)
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
      // { sum = sum + encryptedB; randInt--; };
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

  // String outStr = "Computation finished!";
  func->addStatement(new VarDecl("outStr", "string", new LiteralString("Computation finished!")));

  // printf(outStr);
  func->addStatement(
      new CallExternal("printf",
                       new std::vector<FunctionParameter>(
                           {{"string", new Variable("outStr")}})));

  // return sum;
  func->addStatement(
      new Return(new Variable("sum")));
}

