#include <genAstDemo.h>
#include "main.h"
#include "test/AstTestingGenerator.h"

int main() {
  // runInteractiveDemo();

  // build demo circuit
  Ast ast;
  AstTestingGenerator::generateAst(19, ast);

  // apply cone rewriting to AST
  ConeRewriter coneRewriter(ast);
  coneRewriter.applyConeRewriting();

  return 0;
}


