
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
void generateDemoOne(Function *pFunction) {
    std::cout << "Running generateDemoOne()..." << std::endl;

    // int computePrivate(int x) { ... }
    pFunction->name = "computePrivate";
    FunctionParameter paramX("x", "int");
    pFunction->addParameter(paramX);

    // int a = 4;
    auto literalFour = std::make_unique<LiteralInt>(4);
    auto varDeclA = std::make_unique<VarDecl>("a", "int", std::move(literalFour));
    // accessing a make_unique pointer: varDeclA->datatype << std::endl;
    pFunction->body.push_back(std::move(varDeclA));
    // accessing the VarDeclA in the vec: dynamic_cast<VarDecl*>(funcBody.back().get())->datatype

    // int k;
    auto declareK = std::make_unique<VarDecl>("k", "int", nullptr);
    pFunction->body.push_back(move(declareK));

    // x > 32
    auto thirtyTwo = std::make_unique<LiteralInt>(32);
    auto varX = std::make_unique<Variable>("x");
    auto condition = std::make_unique<LogicalExpr>(std::move(varX), LogicalCompOperator::greater, std::move(thirtyTwo));

    // if ( x > 32 ) { k = x * a; } else { k = (x * a) + 42; }
    // then-branch: { k = x * a; }
    auto varA = std::make_unique<Variable>("a");
    auto varX2 = std::make_unique<Variable>("x");
    auto xTimesA = std::make_unique<BinaryExpr>(std::move(varX2), BinaryOperator::multiplication, std::move(varA));
    auto assignToKIf = std::make_unique<VarAssignm>("k", std::move(xTimesA));
    std::vector<std::unique_ptr<AbstractStatement>> thenStatements;
    thenStatements.push_back(std::move(assignToKIf));
    auto thenBlock = std::make_unique<Block>(std::move(thenStatements));
    // else-branch: else { k = (x * a) + 42; }
    // --- (x * a)
    auto varA2 = std::make_unique<Variable>("a");
    auto varX3 = std::make_unique<Variable>("x");
    auto xTimesA2 = std::make_unique<BinaryExpr>(std::move(varX3), BinaryOperator::multiplication, std::move(varA2));
    std::unique_ptr<AbstractExpr> group = std::make_unique<Group>(std::move(xTimesA2));
    // --- (x * a) + 42
    std::unique_ptr<AbstractExpr> val42 = std::make_unique<LiteralInt>(42);
    auto xTimesAPlus42 = std::make_unique<BinaryExpr>(std::move(group), BinaryOperator::addition, std::move(val42));
    // --- k = ...
    auto assignToKElse = std::make_unique<VarAssignm>("k", std::move(xTimesAPlus42));
    // --- else { ... }
    std::vector<std::unique_ptr<AbstractStatement>> elseStatements;
    elseStatements.push_back(std::move(assignToKElse));
    auto elseBlock = std::make_unique<Block>(move(elseStatements));
    // if-statement
    auto ifStmt = std::make_unique<If>(std::move(condition), std::move(thenBlock), std::move(elseBlock));
    pFunction->body.push_back(move(ifStmt));

    // return k
    auto ret = std::make_unique<Return>(std::make_unique<Variable>("k"));
    pFunction->body.push_back(move(ret));
}

/// Generates an sample AST for the following code:
///
///  \code{.cpp}
///  int determineSuitableX(int encryptedA, int encryptedB) {
///      int randInt = rand() % 42;                  // Call
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
Function generateDemoTwo() {}
//    std::cout << "Running generateDemoTwo()..." << std::endl;
//
//    // statements of the function's body
//    std::vector<std::unique_ptr<AbstractStatement>> funcBody;
//
//    // int randInt = rand() % 42;
//    CallExternal randCall("rand");
//    LiteralInt fourtyTwo(42);
//    BinaryExpr randMod42(&randCall, BinaryOperator::modulo, &fourtyTwo);
//    auto randIntMod42 = std::make_unique<VarDecl>("randInt", "int", &randMod42);
//    funcBody.push_back(std::move(randIntMod42));
//
//    // bool b = encryptedA < 2;
//    Variable varEncA("encryptedA");
//    LiteralInt two(2);
//    LogicalExpr encASmaller2(&varEncA, LogicalCompOperator::smaller, &two);
//    auto declBoolB = std::make_unique<VarDecl>("b", "bool", &encASmaller2);
//    funcBody.push_back(std::move(declBoolB));
//
//    // int sum = 0;
//    LiteralInt null(0);
//    auto declSum = std::make_unique<VarDecl>("sum", "int", &null);
//    funcBody.push_back(std::move(declSum));
//
//    // while (randInt > 0 && !b == true)
//    Block whileBody;
//    // --- randInt > 0
//    Variable varRandInt("randInt");
//    LogicalExpr whileConditionLeft(&varRandInt, LogicalCompOperator::greater, &null);
//    // --- !b == true
//    Variable varB("b");
//    UnaryExpr notB(UnaryOperator::negation, varB);
//    LiteralBool boolTrue(true);
//    LogicalExpr whileConditionRight(&notB, LogicalCompOperator::equal, &boolTrue);
//    // --- (randInt > 0 && !b == true)
//    LogicalExpr whileCondition(&whileConditionLeft, LogicalCompOperator::logicalAnd, &whileConditionRight);
//    // { sum = sum + encryptedB; randInt = randInt-1; }
//    // --- sum = sum + encryptedB;
//    Variable varSum("sum");
//    Variable varEncryptedB("encryptedB");
//    BinaryExpr sumAddEncB(&varSum, BinaryOperator::addition, &varEncryptedB);
//    auto toSum = std::make_unique<VarAssignm>("sum", &sumAddEncB);
//    whileBody.addStatement(std::move(toSum));
//    // --- randInt = randInt-1;
//    Variable randInt("randInt");
//    LiteralInt one(1);
//    BinaryExpr decrementRandInt(&randInt, BinaryOperator::subtraction, &one);
//    auto toRandInt = std::make_unique<VarAssignm>("randInt", &decrementRandInt);
//    whileBody.addStatement(std::move(toRandInt));
//    auto whileLoop = std::make_unique<While>(&whileCondition, &whileBody);
//    funcBody.push_back(std::move(whileLoop));
//
//    // String outStr = "Computation finished!";
//    LiteralString strOut("Computation finished!");
//    auto varStr = std::make_unique<VarAssignm>("outStr", &strOut);
//    funcBody.push_back(std::move(varStr));
//
//    // printf(outStr);
//    std::vector<FunctionParameter> params;
//    params.emplace_back("outStr", "std::string");
//    auto printfCall = std::make_unique<CallExternal>("printf", params);
//    funcBody.push_back(std::move(printfCall));
//
//    // return sum;
//    auto ret = std::make_unique<Return>(&varSum);
//    funcBody.push_back(move(ret));
//
//    // int determineSuitableX(int encryptedA, int encryptedB)
//    Function func("determineSuitableX", move(funcBody));
//    FunctionParameter paramA("encryptedA", "int");
//    func.addParameter(paramA);
//    FunctionParameter paramB("encryptedB", "int");
//    func.addParameter(paramB);
//
//    return func;
//}