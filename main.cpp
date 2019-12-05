#include <iostream>
#include "include/ast/LiteralInt.h"
#include "include/ast/LiteralBool.h"
#include "include/ast/LiteralString.h"
#include "include/ast/VarDecl.h"
#include "include/ast/Variable.h"
#include "include/ast/Block.h"
#include "include/ast/BinaryExpr.h"
#include "include/ast/VarAssignm.h"
#include "include/ast/Group.h"
#include "include/ast/If.h"
#include "include/ast/Return.h"
#include "include/ast/Function.h"
#include "include/ast/FunctionParameter.h"

using namespace std;

/* generateDemo generates an AST for the following code:
 *
 *   int computePrivate(int x) {        // Function
 *      int a = 4;                      // VarDecl, LiteralInt
 *      int k;                          // VarDecl
 *      if (x > 32) {                   // If, Block, Variable
 *          k = x * a;                  // VarAssignm, BinaryExpr, Operator, Variable
 *      } else {                        // Block
 *          k = (x * a) + 42;           // VarAssignm, Group, BinaryExpr, BinaryExpr, Variable
 *      }
 *      return k;                       // Return
 *   }
 *
 *   // Missing
 *   - Call
 *   - Class
 *   - ExpressionStmt
 *   - LiteralBool
 *   - LiteralString
 *   - LogicalExpr
 *   - UnaryExpr
 *   - While
 */
void generateDemo() {

    std::vector<std::unique_ptr<AbstractStatement>> funcBody;

    // int a = 4;
    LiteralInt four(4);
    LiteralBool demoBool(true);
    LiteralString demoString("Blah.");
    auto varDeclA = make_unique<VarDecl>("a", "int", &four);
    // accessing a make_unique pointer
    std::cout << "This works before 'move': " << varDeclA->datatype << std::endl;
    funcBody.push_back(std::move(varDeclA));
    // accessing the VarDeclA after pushing to the vector
    std::cout << dynamic_cast<VarDecl*>(funcBody.back().get())->datatype << std::endl;

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

    cout << "Generation finished.";

    // BACKUP
    // funcBody.emplace_back(std::move(varDeclA));
    // VarDecl *ptr = dynamic_cast<VarDecl*>(funcBody.at(0).get());
}

int main() {
    // generate AST
    cout << "Running generateDemo()...";
    generateDemo();
    cout << "done." << endl;

    // print AST as JSON
    // rootNode.print(format="json");

    // calculate multiplicative depth
    // AstStatistics.calcMultiplicativeDepth(rootNode);

    return 0;
}
