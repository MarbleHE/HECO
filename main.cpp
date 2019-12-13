#include "main.h"
#include "examples/genAstDemo.cpp"
#include "include/visitor/StatisticsVisitor.h"

using namespace std;


int main() {
    std::cout << "Running main()" << std::endl;

//    run();

//    // Test for the visitor pattern
    Ast ast;
    generateDemoOne(ast);
    run();
    auto root = dynamic_cast<Function *>(ast.getRootNode());
    StatisticsVisitor sv;
    sv.visit(*root);

    return 0;
}

