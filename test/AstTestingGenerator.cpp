#include "AstTestingGenerator.h"
#include "Operator.h"
#include "ArithmeticExpr.h"
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
#include "For.h"
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
    {22, AstTestingGenerator::genAstForSecretTaintingWithMultipleNonSequentialStatements},
    {23, AstTestingGenerator::genAstIncludingForStatement}
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
                                 new ArithmeticExpr(
                                     new Variable("inputA"),
                                     ArithmeticOp::multiplication,
                                     new ArithmeticExpr(
                                         new Variable("inputB"),
                                         ArithmeticOp::multiplication,
                                         new Variable("inputC")))));

  // return prod / 3;
  func->addStatement(new Return(new ArithmeticExpr(
      new Variable("prod"),
      ArithmeticOp::division,
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
                                 new ArithmeticExpr(
                                     new Variable("inputA"),
                                     ArithmeticOp::multiplication,
                                     new Variable("inputB"))));

  // prod = prod * inputC
  func->addStatement(
      new VarAssignm("prod", new ArithmeticExpr(
          new Variable("prod"),
          ArithmeticOp::multiplication,
          new Variable("inputC"))));

  // return prod / 3;
  func->addStatement(new Return(new ArithmeticExpr(
      new Variable("prod"),
      ArithmeticOp::division,
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
                                 new ArithmeticExpr(
                                     new Variable("inputA"),
                                     ArithmeticOp::multiplication,
                                     new Variable("inputB"))));

  // int rInt = rand()
  func->addStatement(
      new VarDecl("rInt", Types::INT, new CallExternal("std::rand")));

  // prod = prod * inputC
  func->addStatement(
      new VarAssignm("prod", new ArithmeticExpr(
          new Variable("prod"),
          ArithmeticOp::multiplication,
          new Variable("inputC"))));

  // return prod / 3;
  func->addStatement(new Return(new ArithmeticExpr(
      new Variable("prod"),
      ArithmeticOp::division,
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
                                 new ArithmeticExpr(
                                     new Variable("inputA"),
                                     ArithmeticOp::multiplication,
                                     new Variable("inputB"))));


  // if (prod > 42) { prod = prod * inputC; }
  func->addStatement(new If(
      new LogicalExpr(
          new Variable("prod"), LogCompOp::greater, new LiteralInt(42)),
      new VarAssignm("prod", new ArithmeticExpr(
          new Variable("prod"),
          ArithmeticOp::multiplication,
          new Variable("inputC")))));

  // return prod / 3;
  func->addStatement(new Return(new ArithmeticExpr(
      new Variable("prod"),
      ArithmeticOp::division,
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
                                 new ArithmeticExpr(
                                     new Variable("inputA"),
                                     ArithmeticOp::multiplication,
                                     new Variable("inputB"))));

  // argPow = inputC * inputC
  func->addStatement(
      new VarAssignm("argPow", new ArithmeticExpr(
          new Variable("inputC"),
          ArithmeticOp::multiplication,
          new Variable("inputC"))));

  // return prod / 3;
  func->addStatement(new Return(new ArithmeticExpr(
      new Variable("prod"),
      ArithmeticOp::division,
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
                                 new ArithmeticExpr(
                                     new Variable("inputA"),
                                     ArithmeticOp::multiplication,
                                     new Variable("inputB"))));

  // int prod2 = prod * inputC;
  func->addStatement(new VarDecl("prod2", Types::INT,
                                 new ArithmeticExpr(
                                     new Variable("prod"),
                                     ArithmeticOp::multiplication,
                                     new Variable("inputC"))));

  // return prod / 3;
  func->addStatement(new Return(new ArithmeticExpr(
      new Variable("prod"),
      ArithmeticOp::division,
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
                                 new ArithmeticExpr(
                                     new Variable("width"),
                                     ArithmeticOp::multiplication,
                                     new ArithmeticExpr(
                                         new Variable("length"),
                                         ArithmeticOp::multiplication,
                                         new Variable("depth")))));

  // return prod / 3;
  func->addStatement(new Return(new ArithmeticExpr(
      new Variable("prod"),
      ArithmeticOp::division,
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
                                 new ArithmeticExpr(
                                     new Variable("inputA"),
                                     ArithmeticOp::multiplication,
                                     new Variable("inputB"))));


  // if (takeIf) { prod = prod % 12; } else { prod = prod - 21; }
  func->addStatement(new If(
      new LogicalExpr(
          new Variable("takeIf"), LogCompOp::equal, new LiteralBool(true)),
      new VarAssignm("prod",
                     new ArithmeticExpr(
                         new Variable("prod"),
                         ArithmeticOp::modulo,
                         new LiteralInt(12))),
      new VarAssignm("prod",
                     new ArithmeticExpr(
                         new Variable("prod"),
                         ArithmeticOp::subtraction,
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
          new Variable("strong"), LogCompOp::equal, new LiteralBool(true)),
      new Block(new VarAssignm("inputA",
                               new ArithmeticExpr(
                                   new Variable("inputA"),
                                   ArithmeticOp::multiplication,
                                   new LiteralInt(42)))),
      new Block(new VarAssignm("inputA",
                               new ArithmeticExpr(
                                   new Variable("inputA"),
                                   ArithmeticOp::addition,
                                   new LiteralInt(42))))));

  // if (negate) { inputA = -inputA }
  func->addStatement(new If(
      new LogicalExpr(
          new Variable("negate"), LogCompOp::equal, new LiteralBool(true)),
      new VarAssignm(
          "inputA", new UnaryExpr(UnaryOp::negation, new Variable("inputA")))));

  // return inputA >= inputB
  func->addStatement(
      new Return(
          new LogicalExpr(
              new Variable("inputA"), LogCompOp::greaterEqual, new Variable("inputB"))));

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
          new ArithmeticExpr(new Variable("strA"),
                             ArithmeticOp::addition,
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
                      LogCompOp::smaller,
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
              LogCompOp::greater,
              new LiteralInt(0)),
          LogCompOp::logicalAnd,
          // !b == true
          new LogicalExpr(
              new UnaryExpr(UnaryOp::negation, new Variable("b")),
              LogCompOp::equal,
              new LiteralBool(true))),
      // body: { sum = sum + encryptedB; randInt--; };
      new Block(
          std::vector<AbstractStatement *>{
              // sum = sum + encryptedB
              new VarAssignm("sum", new ArithmeticExpr(
                  new Variable("sum"),
                  ArithmeticOp::addition,
                  new Variable("encryptedB"))),
              // randInt = randInt-1;
              new VarAssignm("randInt", new ArithmeticExpr(
                  new Variable("randInt"),
                  ArithmeticOp::subtraction,
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
                   new Block(new Return(
                       new ArithmeticExpr(
                           new Variable("inputA"),
                           ArithmeticOp::multiplication,
                           new LiteralInt(32)))
                   )))));

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
                             new ArithmeticExpr(new LiteralInt(11), ArithmeticOp::addition, new LiteralInt(213)))},
      new Function("computeSecret",
                   new ParameterList({new FunctionParameter("int", new Variable("inputA"))}),
                   new Block(new Return(
                       new ArithmeticExpr(
                           new Variable("inputA"),
                           ArithmeticOp::multiplication,
                           new LiteralInt(32)))
                   )))));

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
      new LogicalExpr(new Variable("x"), LogCompOp::greater, 32),
      // { k = x * a}
      new Block(
          new VarAssignm("k",
                         new ArithmeticExpr(
                             new Variable("x"),
                             ArithmeticOp::multiplication,
                             new Variable("a")))),
      // else { k = (x * a) + 42; }
      new Block(
          new VarAssignm("k",
                         new ArithmeticExpr(
                             new ArithmeticExpr(
                                 new Variable("x"),
                                 ArithmeticOp::multiplication,
                                 new Variable("a")),
                             ArithmeticOp::addition,
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
                  new ArithmeticExpr(
                      new CallExternal("std::rand"),
                      ArithmeticOp::modulo,
                      new LiteralInt(42))));

  // bool b = encryptedA < 2;
  func->addStatement(
      new VarDecl("b", Types::BOOL,
                  new LogicalExpr(
                      new Variable("encryptedA"),
                      LogCompOp::smaller,
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
              LogCompOp::greater,
              new LiteralInt(0)),
          LogCompOp::logicalAnd,
          // !b == true
          new LogicalExpr(
              new UnaryExpr(UnaryOp::negation, new Variable("b")),
              LogCompOp::equal,
              new LiteralBool(true))),
      // { sum = sum + encryptedB; randInt--; };
      new Block(
          std::vector<AbstractStatement *>{
              // sum = sum + encryptedB
              new VarAssignm("sum", new ArithmeticExpr(
                  new Variable("sum"),
                  ArithmeticOp::addition,
                  new Variable("encryptedB"))),
              // randInt = randInt-1;
              new VarAssignm("randInt", new ArithmeticExpr(
                  new Variable("randInt"),
                  ArithmeticOp::subtraction,
                  new LiteralInt(1)))})));

  // STRING outStr = "Computation finished!";
  func->addStatement(new VarDecl("outStr", "Computation finished!"));

  // int _ = printf(outStr);
  func->addStatement(new VarDecl("_", Types::INT,
                                 new CallExternal("printf", {
                                     new FunctionParameter("string", new Variable("outStr"))})));

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
                                 new ArithmeticExpr(
                                     new Variable("inputA"),
                                     ArithmeticOp::multiplication,
                                     new Variable("inputB"))));

  // prod = prod * inputC
  func->addStatement(
      new VarAssignm("prod", new ArithmeticExpr(
          new Variable("prod"),
          ArithmeticOp::multiplication,
          new Variable("inputC"))));

  // return prod * 3;
  func->addStatement(new Return(new ArithmeticExpr(
      new Variable("prod"),
      ArithmeticOp::multiplication,
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
  func->setParameterList(funcParams);

  // int stdA = 512;
  func->addStatement(new VarDecl("stdA", 512));

  // int stdB = 2 * stdA;
  func->addStatement(
      new VarDecl("stdB",
                  Types::INT,
                  new ArithmeticExpr(new LiteralInt(2),
                                     ArithmeticOp::multiplication,
                                     new Variable("stdA"))));

  // int prod = [base * stdA] + [base * stdB];
  func->addStatement(new VarDecl("prod", Types::INT,
                                 new ArithmeticExpr(
                                     new ArithmeticExpr(
                                         new Variable("base"),
                                         ArithmeticOp::multiplication,
                                         new Variable("stdA")),
                                     ArithmeticOp::addition,
                                     new ArithmeticExpr(
                                         new Variable("base"),
                                         ArithmeticOp::multiplication,
                                         new Variable("stdB")))));

  // int condVal = [22 * defaultC] + [base * useBase];
  func->addStatement(
      new VarDecl("condVal", Types::INT,
                  new ArithmeticExpr(
                      new ArithmeticExpr(
                          new LiteralInt(22),
                          ArithmeticOp::multiplication,
                          new Variable("defaultC")),
                      ArithmeticOp::addition,
                      new ArithmeticExpr(
                          new Variable("base"),
                          ArithmeticOp::multiplication,
                          new Variable("useBase")))));

  // return [prod > 1024] && [condVal >= 112];
  func->addStatement(new Return(
      new LogicalExpr(
          new LogicalExpr(
              new Variable("prod"),
              LogCompOp::greater,
              new LiteralInt(1024)),
          LogCompOp::logicalAnd,
          new LogicalExpr(
              new Variable("condVal"),
              LogCompOp::greaterEqual,
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
                  LogCompOp::logicalAnd,
                  new Variable("a_2^(1)")),
              LogCompOp::logicalXor,
              new LogicalExpr(
                  new Variable("a_1^(2)"),
                  LogCompOp::logicalAnd,
                  new Variable("a_2^(2)"))),
          LogCompOp::logicalXor,
          new Variable("y_1")),
      LogCompOp::logicalAnd,
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
                      LogCompOp::logicalAnd,
                      new Variable("a_1^(1)_right")),
                  LogCompOp::logicalAnd,
                  new LogicalExpr(
                      new Variable("a_2^(1)_left"),
                      LogCompOp::logicalXor,
                      new Variable("a_2^(1)_right"))),
              LogCompOp::logicalXor,
              new LogicalExpr(
                  new LogicalExpr(
                      new Variable("a_1^(2)_left"),
                      LogCompOp::logicalAnd,
                      new Variable("a_1^(2)_right")),
                  LogCompOp::logicalAnd,
                  new LogicalExpr(
                      new Variable("a_2^(2)_left"),
                      LogCompOp::logicalXor,
                      new Variable("a_2^(2)_right")))),
          LogCompOp::logicalXor,
          new Variable("y_1")),
      LogCompOp::logicalAnd,
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
                              LogCompOp::logicalAnd,
                              new Variable("a_1^(1)_right")),
                          LogCompOp::logicalAnd,
                          new LogicalExpr(
                              new Variable("a_2^(1)_left"),
                              LogCompOp::logicalXor,
                              new Variable("a_2^(1)_right"))),

                      LogCompOp::logicalXor,

                      new LogicalExpr(
                          new LogicalExpr(
                              new Variable("a_1^(2)_left"),
                              LogCompOp::logicalAnd,
                              new Variable("a_1^(2)_right")),
                          LogCompOp::logicalAnd,
                          new LogicalExpr(
                              new Variable("a_2^(2)_left"),
                              LogCompOp::logicalXor,
                              new Variable("a_2^(2)_right")))),

                  LogCompOp::logicalXor,
                  new Variable("y_1")),

              LogCompOp::logicalXor,
              new Variable("y_2")),

          LogCompOp::logicalXor,
          new Variable("y_3")),

      LogCompOp::logicalAnd,
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
                                  LogCompOp::logicalAnd,
                                  new Variable("a_1^(1)_right")),
                              LogCompOp::logicalAnd,
                              new LogicalExpr(
                                  new Variable("a_2^(1)_left"),
                                  LogCompOp::logicalXor,
                                  new Variable("a_2^(1)_right"))),

                          LogCompOp::logicalXor,

                          new LogicalExpr(
                              new LogicalExpr(
                                  new Variable("a_1^(2)_left"),
                                  LogCompOp::logicalAnd,
                                  new Variable("a_1^(2)_right")),
                              LogCompOp::logicalAnd,
                              new LogicalExpr(
                                  new Variable("a_2^(2)_left"),
                                  LogCompOp::logicalXor,
                                  new Variable("a_2^(2)_right")))),

                      LogCompOp::logicalXor,
                      new Variable("y_1")),    // level 5

                  LogCompOp::logicalXor,
                  new Variable("y_2")),

              LogCompOp::logicalXor,
              new Variable("y_3")),  // level 3

          LogCompOp::logicalXor,
          new Variable("y_4")),

      LogCompOp::logicalAnd,
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
                                  LogCompOp::logicalXor,
                                  aCone),                 // <-- insertion point of cone 'aCone' from previous stmt
                              LogCompOp::logicalAnd,
                              new Variable("b_1^(1)_right")),
                          LogCompOp::logicalAnd,
                          new LogicalExpr(
                              new Variable("b_2^(1)_left"),
                              LogCompOp::logicalXor,
                              new Variable("b_2^(1)_right"))),

                      LogCompOp::logicalXor,
                      new Variable("z_1")),

                  LogCompOp::logicalXor,
                  new Variable("z_2")),

              LogCompOp::logicalXor,
              new Variable("z_3")),

          LogCompOp::logicalXor,
          new Variable("z_4")),

      LogCompOp::logicalAnd,
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
                  new UnaryExpr(UnaryOp::negation,
                                new LogicalExpr(
                                    new Variable("subtotal"),
                                    LogCompOp::smaller,
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
                               new ArithmeticExpr(
                                   new ArithmeticExpr(new Variable("qualifiesForSpecialDiscount"),
                                                      ArithmeticOp::multiplication,
                                                      new LiteralFloat(0.90)),
                                   ArithmeticOp::addition,
                                   new ArithmeticExpr(
                                       new ArithmeticExpr(new LiteralInt(1),
                                                          ArithmeticOp::subtraction,
                                                          new Variable("qualifiesForSpecialDiscount")),
                                       ArithmeticOp::multiplication,
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
      new Return(new ArithmeticExpr(new Variable("subtotal"),
                                    ArithmeticOp::multiplication,
                                    new Variable("discount"))));
}

void AstTestingGenerator::genAstIncludingForStatement(Ast &ast) {
//    int sum = 0;
//    int base = 2;
//    for (int i = 0; i < 3; i=i+1) {
//      sum = sum + base * i;
//    }
//    return sum

  // int powBase2()
  auto func = new Function("powBase2");
  auto funcParams = new ParameterList();
  funcParams->addChild(
      new FunctionParameter(new Datatype(Types::INT), new Variable("inputA")));
  func->setParameterList(funcParams);

  // int sum = 0;
  func->addStatement(new VarDecl("sum", Types::INT, new LiteralInt(0)));
  // int base = 2;
  func->addStatement(new VarDecl("base", Types::INT, new LiteralInt(2)));

  // int = 0;
  auto forInitializer = new VarDecl("i", Types::INT, new LiteralInt(0));
  // i < 3
  auto forCondition = new LogicalExpr(new Variable("i"), smaller, new LiteralInt(3));
  // i = i+1
  auto forUpdate = new VarAssignm("i", new ArithmeticExpr(new Variable("i"), addition, new LiteralInt(1)));
  // sum = sum + base * i;
  auto forBody = new Block(
      new VarAssignm("sum",
                     new ArithmeticExpr(
                         new Variable("sum"),
                         addition,
                         new ArithmeticExpr(
                             new Variable("base"),
                             multiplication,
                             new Variable("i")))));

  func->addStatement(new For(forInitializer,
                             forCondition,
                             forUpdate,
                             forBody));

  // return sum;
  func->addStatement(new Return(new Variable("sum")));

  ast.setRootNode(func);
}
