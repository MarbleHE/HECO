#include <Ast.h>
#include "main.h"

using namespace std;


int main() {
    // generate AST DemoOne
    cout << "Running generateDemoOne()..." << endl;
    //Ast *astOne;
    //generateDemoOne(astOne);
    generateDemoOne();

    // generate AST for DemoTwo
    cout << "Running generateDemoTwo()..." << endl;
    generateDemoTwo();

    // print AST as JSON
    // rootNode.print(format="json");

    // calculate multiplicative depth
    // AstStatistics.calcMultiplicativeDepth(rootNode);

    return 0;
}


void generateDemoOne() {
    // statements of the function's body
    std::vector<std::unique_ptr<AbstractStatement>> funcBody;

    // int a = 4;
    LiteralInt four(4);
    auto varDeclA = make_unique<VarDecl>("a", "int", &four);
    varDeclA->print();
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
    BinaryExpr bexp(&varX, OperatorType::greater, &thirtyTwo);

    // if ( x > 32 ) { k = x * a; } else { k = (x * a) + 42; }
    // then-branch: { k = x * a; }
    Variable varA("a");
    BinaryExpr xTimesA(&varX, OperatorType::multiplication, &varA);
    auto assignToKIf = make_unique<VarAssignm>("k", &xTimesA);
    std::vector<std::unique_ptr<AbstractStatement>> thenStatements;
    thenStatements.push_back(move(assignToKIf));
    Block thenBlock(&thenStatements);

    // else-branch: else { k = (x * a) + 42; }
    // --- (x * a)
    Group mult(&xTimesA);
    // --- (x * a) + 42
    LiteralInt fourtyTwo(42);
    BinaryExpr xTimesAPlus42(&mult, OperatorType::addition, &fourtyTwo);
    // --- k = ...
    auto assignToKElse = make_unique<VarAssignm>("k", &xTimesAPlus42);
    // --- else { ... }
    std::vector<std::unique_ptr<AbstractStatement>> elseStatements;
    elseStatements.push_back(move(assignToKElse));
    Block elseBlock(&elseStatements);

    // if-statement
    auto ifStmt = make_unique<If>(&bexp, &thenBlock, &elseBlock);
    funcBody.push_back(move(ifStmt));

    // return k
    Variable varK("k");
    auto ret = make_unique<Return>(&varK);
    funcBody.push_back(move(ret));

    // int computePrivate(int x) { ... }
    Function func("computePrivate", move(funcBody));
    FunctionParameter paramX("x", "int");
    func.addParameter(paramX);

    // ast.addRootNode(&func);
}


void generateDemoTwo() {

}

