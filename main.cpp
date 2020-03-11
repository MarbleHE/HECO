#include <OpSymbEnum.h>
#include "genAstDemo.h"
#include "ArithmeticExpr.h"
#include "Variable.h"
#include "main.h"

int main() {
//  runInteractiveDemo();

  auto arithmeticExp = new ArithmeticExpr(new Variable("x"), ADDITION, new LiteralInt(22));
  std::cout << arithmeticExp->toString(true) << std::endl;
  return 0;
}


