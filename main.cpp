#include <genAstDemo.h>
#include <DotPrinter.h>
#include "main.h"
#include "test/AstTestingGenerator.h"

int main() {
  // runInteractiveDemo();

  // build demo circuit
  Ast* ast = new Ast;
  AstTestingGenerator::generateAst(19, *ast);

  std::cout << ast->isReversed() << std::endl;

  // apply cone rewriting to AST
  ConeRewriter coneRewriter(ast);
  auto optimizedAst = coneRewriter.applyConeRewriting();

  return 0;
}


