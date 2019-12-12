#include <Call.h>
#include <iostream>
#include <Function.h>
#include <LiteralInt.h>
#include <LogicalExpr.h>
#include <BinaryExpr.h>
#include <Block.h>
#include <Group.h>
#include <CallExternal.h>
#include <UnaryExpr.h>
#include <LiteralBool.h>
#include <LiteralString.h>
#include <Return.h>
#include <VarAssignm.h>
#include <While.h>
#include <Ast.h>


/// Generates a sample AST for the following code:
///
///  \code{.cpp}
///  int computePrivate(int x) {        // Function
///     int a = 4;                      // VarDecl, LiteralInt
///     int k;                          // VarDecl
///     if (x > 32) {                   // If, Block, Variable
///         k = x * a;                  // VarAssignm, BinaryExpr, Operator, Variable
///     } else {                        // Block
///         k = (x * a) + 42;           // VarAssignm, Group, BinaryExpr, BinaryExpr, Variable
///     }
///     return k;                       // Return
///  }
/// \endcode
///
void generateDemoOne(Ast &ast) {
    std::cout << "Running generateDemoOne()..." << std::endl;

    // int computePrivate(int x) { ... }
    Function *func = dynamic_cast<Function *>(ast.setRootNode(new Function("computePrivate")));
    func->addParameter(FunctionParameter("x", "int"));

    // int a = 4;
    func->addStatement(new VarDecl("x", "int", 4));

    // int k;
    func->addStatement(new VarDecl("x", "int"));

    // if ( x > 32 ) { k = x * a; } else { k = (x * a) + 42; }
    func->addStatement(new If(
            // if (x > 32)
            new LogicalExpr(new Variable("x"), LogicalCompOperator::greater, new LiteralInt(32)),
            // { k = x * a}
            new Block(
                    new VarAssignm("k",
                                   new BinaryExpr(
                                           new Variable("x"),
                                           BinaryOperator::multiplication,
                                           new Variable("a")))
            ),
            // else { k = (x * a) + 42; }
            new Block(
                    new VarAssignm("k",
                                   new BinaryExpr(
                                           new BinaryExpr(
                                                   new Variable("x"),
                                                   BinaryOperator::multiplication,
                                                   new Variable("a")),
                                           BinaryOperator::addition,
                                           new LiteralInt(42))))));

    // return k
    func->addStatement(new Return(new Variable("k")));
}

/// Generates an sample AST for the following code:
///
///  \code{.cpp}
///  int determineSuitableX(int encryptedA, int encryptedB) {
///      int randInt = std::rand() % 42;                  // Call
///      bool b = encryptedA < 2;                    // LiteralBool
///      int sum = 0;                                // LiteralInt
///
///      while (randInt > 0 && !b == true) {         // While, LogicalExpr, UnaryExpr
///          sum = sum + encryptedB;                 // VarAssignm, BinaryExpr
///          randInt--;                              // BinaryExpr
///      };
///
///      String outStr = "Computation finished!";    // LiteralString
///      printf(outStr);
///
///      return sum;
///  }
///  \endcode
void generateDemoTwo(Ast &ast) {
    std::cout << "Running generateDemoTwo()..." << std::endl;

    // int determineSuitableX(int encryptedA, int encryptedB) {...}
    Function *func = dynamic_cast<Function *>(ast.setRootNode(new Function("determineSuitableX")));
    func->addParameter(FunctionParameter("encryptedA", "int"));
    func->addParameter(FunctionParameter("encryptedB", "int"));

    // int randInt = rand() % 42;
    func->addStatement(
            new VarDecl("randInt", "int",
                        new BinaryExpr(
                                new CallExternal("std::rand"),
                                BinaryOperator::modulo,
                                new LiteralInt(42)
                        )));

    // bool b = encryptedA < 2;
    func->addStatement(
            new VarDecl("b", "bool",
                        new LogicalExpr(
                                new Variable("encryptedA"),
                                LogicalCompOperator::smaller,
                                new LiteralInt(2))
            ));

    // int sum = 0;
    func->addStatement(new VarDecl("sum", "int", 0));

    // while (randInt > 0 && !b == true) { sum = sum + encryptedB; randInt--; };
    func->addStatement(new While(
            // (randInt > 0 && !b == true)
            new LogicalExpr(
                    // randInt > 0
                    new LogicalExpr(
                            new Variable("randInt"),
                            LogicalCompOperator::greater,
                            new LiteralInt(0)),
                    LogicalCompOperator::logicalAnd,
                    // !b == true
                    new LogicalExpr(
                            new UnaryExpr(UnaryOperator::negation, new Variable("b")),
                            LogicalCompOperator::equal,
                            new LiteralBool(true))
            ),
            // { sum = sum + encryptedB; randInt--; };
            new Block(
                    new std::vector<AbstractStatement *>{
                            // sum = sum + encryptedB
                            new VarAssignm("sum", new BinaryExpr(new Variable("sum"),
                                                                 BinaryOperator::addition,
                                                                 new Variable("encryptedB"))),
                            // randInt = randInt-1;
                            new VarAssignm("randInt", new BinaryExpr(new Variable("randInt"),
                                                                     BinaryOperator::subtraction,
                                                                     new LiteralInt(1)))}
            )));

    // String outStr = "Computation finished!";
    func->addStatement(new VarDecl("outStr", "string", new LiteralString("Computation finished!")));

    // printf(outStr);
    func->addStatement(
            new CallExternal("printf",
                             new std::vector<FunctionParameter>{FunctionParameter("outStr", "string")}));

    // return sum;
    func->addStatement(
            new Return(new Variable("sum")));
}

void runDemo(void(*func)(Ast &), bool printTree = true) {
    Ast ast;
    func(ast);
    if (printTree) std::cout << *(ast.getRootNode()) << std::endl;
}

void run() {
    runDemo(generateDemoOne);
    runDemo(generateDemoTwo);
}