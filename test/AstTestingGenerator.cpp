#include "AstTestingGenerator.h"
#include "Operator.h"
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
#include "LiteralFloat.h"

static std::map<int, std::function<void(Ast &)> > call = {  /* NOLINT */
    {0, AstTestingGenerator::genSuperSimpleAst},
    {1, AstTestingGenerator::genAstRewritingOne},
    {2, AstTestingGenerator::genAstRewritingTwo},
    {3, AstTestingGenerator::genAstRewritingThree},
    {4, AstTestingGenerator::genAstRewritingFour},
    {5, AstTestingGenerator::genAstRewritingFive},
    {6, AstTestingGenerator::genAstRewritingSix},
    {7, AstTestingGenerator::genAstEvalOne},
    {8, AstTestingGenerator::genAstEvalTwo},
    {9, AstTestingGenerator::genAstEvalThree},
    {10, AstTestingGenerator::genAstEvalFour},
    {11, AstTestingGenerator::genAstEvalFive},
    {12, AstTestingGenerator::genAstEvalSix},
    {13, AstTestingGenerator::genAstEvalSeven},
    {14, AstTestingGenerator::genAstPrintVisitorOne},
    {15, AstTestingGenerator::genAstPrintVisitorTwo},
    {16, AstTestingGenerator::genAstMultDepthOne},
    {17, AstTestingGenerator::genAstMultDepthTwo},
    {18, AstTestingGenerator::genAstRewritingSimple},
    {19, AstTestingGenerator::genAstRewritingSimpleExtended},
    {20, AstTestingGenerator::genAstRewritingMultiInputY},
    {21, AstTestingGenerator::genAstRewritingTwoDepth2ConesButSingleVNode},
    {22, AstTestingGenerator::genAstForSecretTaintingWithMultipleNonSequentialStatements}
};

void AstTestingGenerator::generateAst(int id, Ast &ast) {
  // determine the functions this ID is associated to
  auto it = call.find(id);
  if (it==call.end()) throw std::logic_error("Cannot continue. Invalid id given!");
  // call the function by passing the AST object to be written into
  it->second(ast);
}

size_t AstTestingGenerator::getLargestId() {
  return call.size();
}

void AstTestingGenerator::genSuperSimpleAst(Ast &ast) {
  // float getPiApproximated()
  auto func = new Function("getPiApproximated");
  // return 3.14159265359;
  func->addStatement(new Return(new LiteralFloat(3.14159265359f)));
  ast.setRootNode(func);
}

void AstTestingGenerator::genAstRewritingOne(Ast &ast) {
  // int computePrivate(int inputA, int inputB, int inputC)
  auto func = new Function("computePrivate");
  auto funcParams = new ParameterList();
  funcParams->addChild(
      new FunctionParameter(new Datatype(Types::INT), new Variable("inputA")));
  funcParams->addChild(
      new FunctionParameter(new Datatype(Types::INT), new Variable("inputB")));
  funcParams->addChild(
      new FunctionParameter(new Datatype(Types::INT), new Variable("inputC")));
  func->setParameterList(funcParams);

  // int prod = [inputA * [inputB * inputC]]
  func->addStatement(new VarDecl("prod", Types::INT,
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
  auto funcParams = new ParameterList();
  funcParams->addChild(
      new FunctionParameter(new Datatype(Types::INT), new Variable("inputA")));
  funcParams->addChild(
      new FunctionParameter(new Datatype(Types::INT), new Variable("inputB")));
  funcParams->addChild(
      new FunctionParameter(new Datatype(Types::INT), new Variable("inputC")));
  func->setParameterList(funcParams);

  // int prod = inputA * inputB;
  func->addStatement(new VarDecl("prod", Types::INT,
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
  auto funcParams = new ParameterList();
  funcParams->addChild(
      new FunctionParameter(new Datatype(Types::INT), new Variable("inputA")));
  funcParams->addChild(
      new FunctionParameter(new Datatype(Types::INT), new Variable("inputB")));
  funcParams->addChild(
      new FunctionParameter(new Datatype(Types::INT), new Variable("inputC")));
  func->setParameterList(funcParams);

  // int prod = inputA * inputB;
  func->addStatement(new VarDecl("prod", Types::INT,
                                 new BinaryExpr(
                                     new Variable("inputA"),
                                     OpSymb::multiplication,
                                     new Variable("inputB"))));

  // int rInt = rand()
  func->addStatement(
      new VarDecl("rInt", Types::INT, new CallExternal("std::rand")));

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
  auto funcParams = new ParameterList();
  funcParams->addChild(
      new FunctionParameter(new Datatype(Types::INT), new Variable("inputA")));
  funcParams->addChild(
      new FunctionParameter(new Datatype(Types::INT), new Variable("inputB")));
  funcParams->addChild(
      new FunctionParameter(new Datatype(Types::INT), new Variable("inputC")));
  func->setParameterList(funcParams);

  // int prod = inputA * inputB;
  func->addStatement(new VarDecl("prod", Types::INT,
                                 new BinaryExpr(
                                     new Variable("inputA"),
                                     OpSymb::multiplication,
                                     new Variable("inputB"))));


  // if (prod > 42) { prod = prod * inputC; }
  func->addStatement(new If(
      new LogicalExpr(
          new Variable("prod"), OpSymb::greater, new LiteralInt(42)),
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
  auto funcParams = new ParameterList();
  funcParams->addChild(new FunctionParameter(
      new Datatype(Types::INT), new Variable("inputA")));
  funcParams->addChild(new FunctionParameter(
      new Datatype(Types::INT), new Variable("inputB")));
  funcParams->addChild(new FunctionParameter(
      new Datatype(Types::INT), new Variable("inputC")));
  func->setParameterList(funcParams);

  // int prod = inputA * inputB;
  func->addStatement(new VarDecl("prod", Types::INT,
                                 new BinaryExpr(
                                     new Variable("inputA"),
                                     OpSymb::multiplication,
                                     new Variable("inputB"))));

  // argPow = inputC * inputC
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
  auto funcParams = new ParameterList();
  funcParams->addChild(new FunctionParameter(
      new Datatype(Types::INT), new Variable("inputA")));
  funcParams->addChild(new FunctionParameter(
      new Datatype(Types::INT), new Variable("inputB")));
  funcParams->addChild(new FunctionParameter(
      new Datatype(Types::INT), new Variable("inputC")));
  func->setParameterList(funcParams);

  // int prod = inputA * inputB;
  func->addStatement(new VarDecl("prod", Types::INT,
                                 new BinaryExpr(
                                     new Variable("inputA"),
                                     OpSymb::multiplication,
                                     new Variable("inputB"))));

  // int prod2 = prod * inputC;
  func->addStatement(new VarDecl("prod2", Types::INT,
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
  auto funcParams = new ParameterList();
  funcParams->addChild(new FunctionParameter(
      new Datatype(Types::INT), new Variable("width")));
  funcParams->addChild(new FunctionParameter(
      new Datatype(Types::INT), new Variable("length")));
  funcParams->addChild(new FunctionParameter(
      new Datatype(Types::INT), new Variable("depth")));
  func->setParameterList(funcParams);

  // int prod = inputA * inputB * inputC
  func->addStatement(new VarDecl("prod", Types::INT,
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
  auto funcParams = new ParameterList();
  funcParams->addChild(new FunctionParameter(
      new Datatype(Types::INT), new Variable("inputA")));
  funcParams->addChild(new FunctionParameter(
      new Datatype(Types::INT), new Variable("inputB")));
  funcParams->addChild(new FunctionParameter(
      new Datatype(Types::BOOL), new Variable("takeIf")));
  func->setParameterList(funcParams);

  // int prod = inputA * inputB;
  func->addStatement(new VarDecl("prod", Types::INT,
                                 new BinaryExpr(
                                     new Variable("inputA"),
                                     OpSymb::multiplication,
                                     new Variable("inputB"))));


  // if (takeIf) { prod = prod % 12; } else { prod = prod - 21; }
  func->addStatement(new If(
      new LogicalExpr(
          new Variable("takeIf"), OpSymb::equal, new LiteralBool(true)),
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
  auto funcParams = new ParameterList();
  funcParams->addChild(new FunctionParameter(
      new Datatype(Types::INT), new Variable("inputA")));
  funcParams->addChild(new FunctionParameter(
      new Datatype(Types::BOOL), new Variable("strong")));
  funcParams->addChild(new FunctionParameter(
      new Datatype(Types::BOOL), new Variable("negate")));
  funcParams->addChild(new FunctionParameter(
      new Datatype(Types::INT), new Variable("inputB")));
  func->setParameterList(funcParams);

  // if (strong == true) { inputA = inputA * 42; } else { inputA = inputA + 42; }
  func->addStatement(new If(
      new LogicalExpr(
          new Variable("strong"), OpSymb::equal, new LiteralBool(true)),
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
      new LogicalExpr(
          new Variable("negate"), OpSymb::equal, new LiteralBool(true)),
      new VarAssignm(
          "inputA", new UnaryExpr(OpSymb::negation, new Variable("inputA")))));

  // return inputA >= inputB
  func->addStatement(
      new Return(
          new LogicalExpr(
              new Variable("inputA"), OpSymb::greaterEqual, new Variable("inputB"))));

  ast.setRootNode(func);
}

void AstTestingGenerator::genAstEvalFour(Ast &ast) {
  auto func = new Function("concatString");
  auto funcParams = new ParameterList();
  funcParams->addChild(
      new FunctionParameter(new Datatype(Types::STRING), new Variable("strA")));
  funcParams->addChild(
      new FunctionParameter(new Datatype(Types::STRING), new Variable("strB")));
  func->setParameterList(funcParams);

  // return strA + strB
  func->addStatement(
      new Return(
          new BinaryExpr(new Variable("strA"),
                         OpSymb::addition,
                         new Variable("strB"))));

  ast.setRootNode(func);
}

void AstTestingGenerator::genAstEvalFive(Ast &ast) {
  // int determineSuitableX(int encryptedA, int encryptedB) {...}
  Function *func = dynamic_cast<Function *>(ast.setRootNode(new Function("determineSuitableX")));
  func->addParameter(
      new FunctionParameter("int", new Variable("encryptedA")));
  func->addParameter(
      new FunctionParameter("int", new Variable("encryptedB")));
  func->addParameter(
      new FunctionParameter("int", new Variable("randInt")));

  // bool b = encryptedA < 2;
  func->addStatement(
      new VarDecl("b", Types::BOOL,
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
          new std::vector<AbstractStatement *>{
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
  auto *fnc = new Function("nestedCall");
  ast.setRootNode(fnc);

  // int result = computeSecret(33);
  // -> computeSecret(int inputA) { return inputA * 32 }
  fnc->addStatement(new VarAssignm("result", new Call(
      {new FunctionParameter("int", new LiteralInt(33))},
      new Function("computeSecret",
                   new ParameterList({new FunctionParameter("int", new Variable("inputA"))}),
                   new Block({new Return(
                       new BinaryExpr(
                           new Variable("inputA"),
                           OpSymb::multiplication,
                           new LiteralInt(32)))
                   })))));

  // return result;
  fnc->addStatement(new Return(new Variable("result")));
}

void AstTestingGenerator::genAstEvalSeven(Ast &ast) {
  // int nestedCall() { ...
  auto *fnc = new Function("nestedCall");
  ast.setRootNode(fnc);

  // int result = computeSecret(33);
  // -> computeSecret(int inputA) { return inputA * 32 }
  fnc->addStatement(new VarAssignm("result", new Call(
      {new FunctionParameter("int",
                             new BinaryExpr(new LiteralInt(11), OpSymb::addition, new LiteralInt(213)))},
      new Function("computeSecret",
                   new ParameterList({new FunctionParameter("int", new Variable("inputA"))}),
                   new Block({new Return(
                       new BinaryExpr(
                           new Variable("inputA"),
                           OpSymb::multiplication,
                           new LiteralInt(32)))
                   })))));

  // return result;
  fnc->addStatement(new Return(new Variable("result")));
}

void AstTestingGenerator::genAstPrintVisitorOne(Ast &ast) {
  // int computePrivate(int x) { ... }
  Function *func = dynamic_cast<Function *>(ast.setRootNode(new Function("computePrivate")));
  func->addParameter(new FunctionParameter("int", new Variable("x")));

  // int a = 4;
  func->addStatement(new VarDecl("a", 4));

  // int k;
  func->addStatement(new VarDecl("k", Types::INT, nullptr));

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
                             new BinaryExpr(
                                 new Variable("x"),
                                 OpSymb::BinaryOp::multiplication,
                                 new Variable("a")),
                             OpSymb::BinaryOp::addition,
                             42)))));

  // return k
  func->addStatement(new Return(new Variable("k")));
}

void AstTestingGenerator::genAstPrintVisitorTwo(Ast &ast) {
// int determineSuitableX(int encryptedA, int encryptedB) {...}
  Function *func = dynamic_cast<Function *>(ast.setRootNode(new Function("determineSuitableX")));
  func->addParameter(new FunctionParameter("int", new Variable("encryptedA")));
  func->addParameter(new FunctionParameter("int", new Variable("encryptedB")));

  // int randInt = rand() % 42;
  func->addStatement(
      new VarDecl("randInt", Types::INT,
                  new BinaryExpr(
                      new CallExternal("std::rand"),
                      OpSymb::BinaryOp::modulo,
                      new LiteralInt(42))));

  // bool b = encryptedA < 2;
  func->addStatement(
      new VarDecl("b", Types::BOOL,
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
          new std::vector<AbstractStatement *>{
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

  // STRING outStr = "Computation finished!";
  func->addStatement(new VarDecl("outStr", "Computation finished!"));

  // printf(outStr);
  func->addStatement(
      new CallExternal("printf", {
          new FunctionParameter("string", new Variable("outStr"))}));

  // return sum;
  func->addStatement(
      new Return(new Variable("sum")));
}

void AstTestingGenerator::genAstMultDepthOne(Ast &ast) {
  // int computePrivate(int inputA, int inputB, int inputC)
  auto func = new Function("computePrivate");
  auto funcParams = new ParameterList();
  funcParams->addChild(
      new FunctionParameter(new Datatype(Types::INT), new Variable("inputA")));
  funcParams->addChild(
      new FunctionParameter(new Datatype(Types::INT), new Variable("inputB")));
  funcParams->addChild(
      new FunctionParameter(new Datatype(Types::INT), new Variable("inputC")));
  func->setParameterList(funcParams);

  // int prod = inputA * inputB;
  func->addStatement(new VarDecl("prod", Types::INT,
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

  // return prod * 3;
  func->addStatement(new Return(new BinaryExpr(
      new Variable("prod"),
      OpSymb::multiplication,
      new LiteralInt(3))));

  ast.setRootNode(func);
}

void AstTestingGenerator::genAstMultDepthTwo(Ast &ast) {
  // int computePrivate(int inputA, int inputB, int inputC)
  auto func = new Function("computePrivate");
  auto funcParams = new ParameterList();
  funcParams->addChild(new FunctionParameter(
      new Datatype(Types::INT), new Variable("base")));
  funcParams->addChild(new FunctionParameter(
      new Datatype(Types::INT), new Variable("defaultC")));
  funcParams->addChild(new FunctionParameter(
      new Datatype(Types::BOOL), new Variable("useBase")));

  // int stdA = 512;
  func->addStatement(new VarDecl("stdA", 512));

  // int stdB = 2 * stdA;
  func->addStatement(
      new VarDecl("stdB",
                  Types::INT,
                  new BinaryExpr(new LiteralInt(2),
                                 OpSymb::multiplication,
                                 new Variable("stdA"))));

  // int prod = [base * stdA] + [base * stdB];
  func->addStatement(new VarDecl("prod", Types::INT,
                                 new BinaryExpr(
                                     new BinaryExpr(
                                         new Variable("base"),
                                         OpSymb::multiplication,
                                         new Variable("stdA")),
                                     OpSymb::addition,
                                     new BinaryExpr(
                                         new Variable("base"),
                                         OpSymb::multiplication,
                                         new Variable("stdB")))));

  // int condVal = [22 * defaultC] + [base * useBase];
  func->addStatement(
      new VarDecl("condVal", Types::INT,
                  new BinaryExpr(
                      new BinaryExpr(
                          new LiteralInt(22),
                          OpSymb::multiplication,
                          new Variable("defaultC")),
                      OpSymb::addition,
                      new BinaryExpr(
                          new Variable("base"),
                          OpSymb::multiplication,
                          new Variable("useBase")))));

  // return [prod > 1024] && [condVal >= 112];
  func->addStatement(new Return(
      new LogicalExpr(
          new LogicalExpr(
              new Variable("prod"),
              OpSymb::greater,
              new LiteralInt(1024)),
          OpSymb::logicalAnd,
          new LogicalExpr(
              new Variable("condVal"),
              OpSymb::greaterEqual,
              new LiteralInt(112)))));

  ast.setRootNode(func);
}

void AstTestingGenerator::genAstRewritingSimple(Ast &ast) {
  // -----------------------------
  // Schematic diagram of the AST
  // -----------------------------
  // ┌───────┐   ┌───────┐   ┌───────┐   ┌───────┐
  // │a_1^(1)│   │a_2^(1)│   │a_1^(2)│   │a_2^(2)│
  // └───────┘   └───────┘   └───────┘   └───────┘
  //    ▲    .─.    ▲           ▲    .─.    ▲
  //    └───( & )───┘           └───( & )───┘
  //         `─'                     `─'
  //          ▲          .─.          ▲
  //          └─────────( + )─────────┘
  //                     `─'
  //                      ▲
  //                      │     .─.     ┌───────┐
  //                      └────( + )───▶│  y_1  │
  //                            `─'     └───────┘
  //                             ▲
  //                             │     .─.      ┌───────┐
  //                             └────( & )────▶│  a_t  │
  //                                   `─'      └───────┘
  //                                    ▲
  //                                    │
  //                               ┌─────────┐
  //                               │ return  │
  //                               └─────────┘
  auto *returnStatement = new Return(new LogicalExpr(
      new LogicalExpr(
          new LogicalExpr(
              new LogicalExpr(
                  new Variable("a_1^(1)"),
                  OpSymb::logicalAnd,
                  new Variable("a_2^(1)")),
              OpSymb::logicalXor,
              new LogicalExpr(
                  new Variable("a_1^(2)"),
                  OpSymb::logicalAnd,
                  new Variable("a_2^(2)"))),
          OpSymb::logicalXor,
          new Variable("y_1")),
      OpSymb::logicalAnd,
      new Variable("a_t")));
  ast.setRootNode(returnStatement);
}

void AstTestingGenerator::genAstRewritingSimpleExtended(Ast &ast) {
  // -----------------------------
  // Schematic diagram of the AST
  // -----------------------------
  // ┌────────────┐    ┌─────────────┐ ┌────────────┐     ┌─────────────┐     ┌────────────┐    ┌─────────────┐ ┌────────────┐     ┌─────────────┐
  // │a_1^(1)_left│    │a_1^(1)_right│ │a_1^(2)_left│     │a_1^(2)_right│     │a_1^(2)_left│    │a_1^(2)_right│ │a_2^(2)_left│     │a_2^(2)_right│
  // └────────────┘    └─────────────┘ └────────────┘     └─────────────┘     └────────────┘    └─────────────┘ └────────────┘     └─────────────┘
  //       ▲                 ▲               ▲                  ▲                   ▲                 ▲               ▲                  ▲
  //       │       .─.       │               │        .─.       │                   │       .─.       │               │        .─.       │
  //       └──────( & )──────┘               └───────( + )──────┘                   └──────( & )──────┘               └───────( + )──────┘
  //               `─'                                `─'                                   `─'                                `─'
  //                ▲                                  ▲                                     ▲                                  ▲
  //                │               .─.                │                                     │                .─.               │
  //                └──────────────( & )───────────────┘                                     └───────────────( & )──────────────┘
  //                                `─'                                                                       `─'
  //                                 ▲                                                                         ▲
  //                                 │                              .─.                                        │
  //                                 └─────────────────────────────( + )───────────────────────────────────────┘
  //                                                                `─'
  //                                                                 ▲
  //                                                                 │     .─.     ┌───────┐
  //                                                                 └────( + )───▶│  y_1  │
  //                                                                       `─'     └───────┘
  //                                                                        ▲
  //                                                                        │     .─.      ┌───────┐
  //                                                                        └────( & )────▶│  a_t  │
  //                                                                              `─'      └───────┘
  //                                                                               ▲
  //                                                                               │
  //                                                                          ┌─────────┐
  //                                                                          │ return  │
  //                                                                          └─────────┘
  auto *returnStatement = new Return(new LogicalExpr(
      new LogicalExpr(
          new LogicalExpr(
              new LogicalExpr(
                  new LogicalExpr(
                      new Variable("a_1^(1)_left"),
                      OpSymb::logicalAnd,
                      new Variable("a_1^(1)_right")),
                  OpSymb::logicalAnd,
                  new LogicalExpr(
                      new Variable("a_2^(1)_left"),
                      OpSymb::logicalXor,
                      new Variable("a_2^(1)_right"))),
              OpSymb::logicalXor,
              new LogicalExpr(
                  new LogicalExpr(
                      new Variable("a_1^(2)_left"),
                      OpSymb::logicalAnd,
                      new Variable("a_1^(2)_right")),
                  OpSymb::logicalAnd,
                  new LogicalExpr(
                      new Variable("a_2^(2)_left"),
                      OpSymb::logicalXor,
                      new Variable("a_2^(2)_right")))),
          OpSymb::logicalXor,
          new Variable("y_1")),
      OpSymb::logicalAnd,
      new Variable("a_t")));
  ast.setRootNode(returnStatement);
}

void AstTestingGenerator::genAstRewritingMultiInputY(Ast &ast) {
  auto *returnStatement = new Return(new LogicalExpr(
      new LogicalExpr(

          new LogicalExpr(

              new LogicalExpr(

                  new LogicalExpr(

                      new LogicalExpr(
                          new LogicalExpr(
                              new Variable("a_1^(1)_left"),
                              OpSymb::logicalAnd,
                              new Variable("a_1^(1)_right")),
                          OpSymb::logicalAnd,
                          new LogicalExpr(
                              new Variable("a_2^(1)_left"),
                              OpSymb::logicalXor,
                              new Variable("a_2^(1)_right"))),

                      OpSymb::logicalXor,

                      new LogicalExpr(
                          new LogicalExpr(
                              new Variable("a_1^(2)_left"),
                              OpSymb::logicalAnd,
                              new Variable("a_1^(2)_right")),
                          OpSymb::logicalAnd,
                          new LogicalExpr(
                              new Variable("a_2^(2)_left"),
                              OpSymb::logicalXor,
                              new Variable("a_2^(2)_right")))),

                  OpSymb::logicalXor,
                  new Variable("y_1")),

              OpSymb::logicalXor,
              new Variable("y_2")),

          OpSymb::logicalXor,
          new Variable("y_3")),

      OpSymb::logicalAnd,
      new Variable("a_t")));

  ast.setRootNode(returnStatement);
}

void AstTestingGenerator::genAstRewritingTwoDepth2ConesButSingleVNode(Ast &ast) {
  AbstractExpr *aCone = new LogicalExpr(  // level 1
      new LogicalExpr(

          new LogicalExpr(  // level 3

              new LogicalExpr(

                  new LogicalExpr(  // level 5

                      new LogicalExpr(

                          new LogicalExpr(
                              new LogicalExpr(
                                  new Variable("a_1^(1)_left"),
                                  OpSymb::logicalAnd,
                                  new Variable("a_1^(1)_right")),
                              OpSymb::logicalAnd,
                              new LogicalExpr(
                                  new Variable("a_2^(1)_left"),
                                  OpSymb::logicalXor,
                                  new Variable("a_2^(1)_right"))),

                          OpSymb::logicalXor,

                          new LogicalExpr(
                              new LogicalExpr(
                                  new Variable("a_1^(2)_left"),
                                  OpSymb::logicalAnd,
                                  new Variable("a_1^(2)_right")),
                              OpSymb::logicalAnd,
                              new LogicalExpr(
                                  new Variable("a_2^(2)_left"),
                                  OpSymb::logicalXor,
                                  new Variable("a_2^(2)_right")))),

                      OpSymb::logicalXor,
                      new Variable("y_1")),    // level 5

                  OpSymb::logicalXor,
                  new Variable("y_2")),

              OpSymb::logicalXor,
              new Variable("y_3")),  // level 3

          OpSymb::logicalXor,
          new Variable("y_4")),

      OpSymb::logicalAnd,
      new Variable("a_t"));  // level 1

  auto *returnStatement = new Return(new LogicalExpr(
      new LogicalExpr(

          new LogicalExpr(

              new LogicalExpr(

                  new LogicalExpr(

                      new LogicalExpr(
                          new LogicalExpr(
                              new LogicalExpr(
                                  new Variable("b_1^(1)_left"),
                                  OpSymb::logicalXor,
                                  aCone),                 // <-- insertion point of cone 'aCone' from previous stmt
                              OpSymb::logicalAnd,
                              new Variable("b_1^(1)_right")),
                          OpSymb::logicalAnd,
                          new LogicalExpr(
                              new Variable("b_2^(1)_left"),
                              OpSymb::logicalXor,
                              new Variable("b_2^(1)_right"))),

                      OpSymb::logicalXor,
                      new Variable("z_1")),

                  OpSymb::logicalXor,
                  new Variable("z_2")),

              OpSymb::logicalXor,
              new Variable("z_3")),

          OpSymb::logicalXor,
          new Variable("z_4")),

      OpSymb::logicalAnd,
      new Variable("b_t")));

  ast.setRootNode(returnStatement);
}

/// This AST represents the following program:
/// \code{.cpp}
///   float computeTotal(int subtotal) {
///     bool qualifiesForSpecialDiscount = !(subtotal < 1000);
///     secret_float discount = computeDiscountOnServer(secret_bool qualifiesForSpecialDiscount, secret_float subtotal)
///     return subtotal*discount;
///   }
///
///   float computeDiscountOnServer(secret_bool qualifiesForSpecialDiscount) {
///     secret_float discountRate;
///     { // this Block "{...}"does not make any sense but is valid and used for testing
///       discountRate = qualifiesForSpecialDiscount * 0.90 + (1-qualifiesForSpecialDiscount) * 0.98;
///     }
///     return discountRate;
///   }
/// \endcode
void AstTestingGenerator::genAstForSecretTaintingWithMultipleNonSequentialStatements(Ast &ast) {
// int determineSuitableX(int encryptedA, int encryptedB) {...}
  auto funcComputeTotal = dynamic_cast<Function *>(ast.setRootNode(new Function("computeTotal")));
  funcComputeTotal->addParameter(new FunctionParameter(new Datatype(Types::INT, false), new Variable("subtotal")));

  // bool qualifiesForSpecialDiscount = !(subtotal < 1000);
  funcComputeTotal->addStatement(
      new VarDecl("qualifiesForSpecialDiscount",
                  new Datatype(Types::BOOL, false),
                  new UnaryExpr(OpSymb::negation,
                                new LogicalExpr(
                                    new Variable("subtotal"),
                                    OpSymb::smaller,
                                    new LiteralInt(1'000)))));

  // float computeDiscountOnServer(secret_bool qualifiesForSpecialDiscount) {
  auto funcComputeDiscountOnServer = new Function("computeDiscountOnServer");
  funcComputeDiscountOnServer->addParameter(new FunctionParameter(
      new Datatype(Types::BOOL, true),
      new Variable("qualifiesForSpecialDiscount")));

  //  secret_float discountRate;
  funcComputeDiscountOnServer->addStatement(
      new VarDecl("discountRate", new Datatype(Types::FLOAT, true), new LiteralFloat(0.0f)));

  //  { discountRate = qualifiesForSpecialDiscount * 0.90 + (1-qualifiesForSpecialDiscount) * 0.98; }
  funcComputeDiscountOnServer->addStatement(
      new Block(new VarAssignm("discountRate",
                               new BinaryExpr(
                                   new BinaryExpr(new Variable("qualifiesForSpecialDiscount"),
                                                  OpSymb::multiplication,
                                                  new LiteralFloat(0.90)),
                                   OpSymb::addition,
                                   new BinaryExpr(
                                       new BinaryExpr(new LiteralInt(1),
                                                      OpSymb::subtraction,
                                                      new Variable("qualifiesForSpecialDiscount")),
                                       OpSymb::multiplication,
                                       new LiteralFloat(0.98))))));

  //  return discountRate;
  funcComputeDiscountOnServer->addStatement(new Return(new Variable("discountRate")));

  // secret_float discount = computeDiscountOnServer(secret_bool qualifiesForSpecialDiscount)
  funcComputeTotal->addStatement(new VarDecl("discount",
                                             new Datatype(Types::FLOAT, true),
                                             new Call({new FunctionParameter(new Datatype(Types::BOOL, true),
                                                                             new Variable("qualifiesForSpecialDiscount"))},
                                                      funcComputeDiscountOnServer)));

  // return subtotal*discount;
  funcComputeTotal->addStatement(
      new Return(new BinaryExpr(new Variable("subtotal"),
                                OpSymb::multiplication,
                                new Variable("discount"))));
}
