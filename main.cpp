#include <Call.h>
#include "main.h"
#include "examples/genAstDemo.cpp"
#include <sstream>

using namespace std;


int main() {
    // generate AST of DemoOne
    Function func;
    generateDemoOne(&func);

    json j = func;
    cout << j.dump() << std::endl;
    cout << "test" << std::endl;


    // generate AST for DemoTwo
    //auto rootNodeTwo = generateDemoTwo();

    // print AST as JSON
    // rootNodeOne.print(format="json");

    // calculate multiplicative depth
    // AstStatistics.calcMultiplicativeDepth(rootNode);

    return 0;
}


