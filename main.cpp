#include <genAstDemo.h>
#include <DotPrinter.h>
#include "main.h"
#include "test/AstTestingGenerator.h"

int main() {
  // runInteractiveDemo();

  // build demo circuit
  Ast* ast = new Ast;
  AstTestingGenerator::generateAst(19, *ast);

  // apply cone rewriting to AST
  ConeRewriter coneRewriter(ast);
  auto optimizedAst = coneRewriter.applyConeRewriting();

  return 0;
}


