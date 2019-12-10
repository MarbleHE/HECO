#include <Call.h>
#include "main.h"
#include <experimental/type_traits>
#include <sstream>

using namespace std;


int main() {
    // generate AST for DemoOne
    //đauto rootNodeOne = generateDemoOne();

    LiteralInt lint(312);
    std::cout << lint << std::endl;

    // generate AST for DemoTwo
    auto rootNodeTwo = generateDemoTwo();

    // print AST as JSON
    // rootNodeOne.print(format="json");

    // calculate multiplicative depth
    // AstStatistics.calcMultiplicativeDepth(rootNode);

    return 0;
}


//void generateDemoOne(Ast *ast) {
Function generateDemoOne() {
    cout << "Running generateDemoOne()..." << endl;

    // statements of the function's body
    std::vector<std::unique_ptr<AbstractStatement>> funcBody;

    // int a = 4;
    LiteralInt four(4);
    auto varDeclA = make_unique<VarDecl>("a", "int", &four);
    // accessing a make_unique pointer
    //std::cout << "This works before 'move': " << varDeclA->datatype << std::endl;
    funcBody.push_back(std::move(varDeclA));
    // accessing the VarDeclA after pushing to the vector
    //std::cout << dynamic_cast<VarDecl*>(funcBody.back().get())->datatype << std::endl;

    // int k;
    auto declareK = make_unique<VarDecl>("k", "int", nullptr);
    funcBody.push_back(move(declareK));

    // x > 32
    LiteralInt thirtyTwo(32);
    Variable varX("x");
    LogicalExpr lexp(&varX, LogicalCompOperator::greater, &thirtyTwo);

    // if ( x > 32 ) { k = x * a; } else { k = (x * a) + 42; }
    // then-branch: { k = x * a; }
    Variable varA("a");
    BinaryExpr xTimesA(&varX, BinaryOperator::multiplication, &varA);
    auto assignToKIf = make_unique<VarAssignm>("k", &xTimesA);
    Block thenBlock;
    thenBlock.addStatement(move(assignToKIf));

    // else-branch: else { k = (x * a) + 42; }
    // --- (x * a)
    Group mult(&xTimesA);
    // --- (x * a) + 42
    LiteralInt fourtyTwo(42);
    BinaryExpr xTimesAPlus42(&mult, BinaryOperator::addition, &fourtyTwo);
    // --- k = ...
    auto assignToKElse = make_unique<VarAssignm>("k", &xTimesAPlus42);
    // --- else { ... }
    Block elseBlock;
    elseBlock.addStatement(move(assignToKElse));

    // if-statement
    auto ifStmt = make_unique<If>(&lexp, &thenBlock, &elseBlock);
    funcBody.push_back(move(ifStmt));

    // return k
    Variable varK("k");
    auto ret = make_unique<Return>(&varK);
    funcBody.push_back(move(ret));

    // int computePrivate(int x) { ... }
    Function func("computePrivate", move(funcBody));
    FunctionParameter paramX("x", "int");
    func.addParameter(paramX);

    return func;
}


Function generateDemoTwo() {
    cout << "Running generateDemoTwo()..." << endl;

    // statements of the function's body
    std::vector<std::unique_ptr<AbstractStatement>> funcBody;

    // int randInt = rand() % 42;
    CallExternal randCall("rand");
    LiteralInt fourtyTwo(42);
    BinaryExpr randMod42(&randCall, BinaryOperator::modulo, &fourtyTwo);
    auto randIntMod42 = make_unique<VarDecl>("randInt", "int", &randMod42);
    funcBody.push_back(std::move(randIntMod42));

    // bool b = encryptedA < 2;
    Variable varEncA("encryptedA");
    LiteralInt two(2);
    LogicalExpr encASmaller2(&varEncA, LogicalCompOperator::smaller, &two);
    auto declBoolB = make_unique<VarDecl>("b", "bool", &encASmaller2);
    funcBody.push_back(std::move(declBoolB));

    // int sum = 0;
    LiteralInt null(0);
    auto declSum = make_unique<VarDecl>("sum", "int", &null);
    funcBody.push_back(std::move(declSum));

    // while (randInt > 0 && !b == true)
    Block whileBody;
    // --- randInt > 0
    Variable varRandInt("randInt");
    LogicalExpr whileConditionLeft(&varRandInt, LogicalCompOperator::greater, &null);
    // --- !b == true
    Variable varB("b");
    UnaryExpr notB(UnaryOperator::negation, varB);
    LiteralBool boolTrue(true);
    LogicalExpr whileConditionRight(&notB, LogicalCompOperator::equal, &boolTrue);
    // --- (randInt > 0 && !b == true)
    LogicalExpr whileCondition(&whileConditionLeft, LogicalCompOperator::logicalAnd, &whileConditionRight);
    // { sum = sum + encryptedB; randInt = randInt-1; }
    // --- sum = sum + encryptedB;
    Variable varSum("sum");
    Variable varEncryptedB("encryptedB");
    BinaryExpr sumAddEncB(&varSum, BinaryOperator::addition, &varEncryptedB);
    auto toSum = make_unique<VarAssignm>("sum", &sumAddEncB);
    whileBody.addStatement(std::move(toSum));
    // --- randInt = randInt-1;
    Variable randInt("randInt");
    LiteralInt one(1);
    BinaryExpr decrementRandInt(&randInt, BinaryOperator::subtraction, &one);
    auto toRandInt = make_unique<VarAssignm>("randInt", &decrementRandInt);
    whileBody.addStatement(std::move(toRandInt));
    auto whileLoop = make_unique<While>(&whileCondition, &whileBody);
    funcBody.push_back(std::move(whileLoop));

    // String outStr = "Computation finished!";
    LiteralString strOut("Computation finished!");
    auto varStr = make_unique<VarAssignm>("outStr", &strOut);
    funcBody.push_back(std::move(varStr));

    // printf(outStr);
    vector<FunctionParameter> params;
    params.push_back(FunctionParameter("outStr", "std::string"));
    auto printfCall = make_unique<CallExternal>("printf", params);
    funcBody.push_back(std::move(printfCall));

    // return sum;
    auto ret = make_unique<Return>(&varSum);
    funcBody.push_back(move(ret));

    // int determineSuitableX(int encryptedA, int encryptedB)
    Function func("determineSuitableX", move(funcBody));
    FunctionParameter paramA("encryptedA", "int");
    func.addParameter(paramA);
    FunctionParameter paramB("encryptedB", "int");
    func.addParameter(paramB);

    return func;
}

