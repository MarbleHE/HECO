#include "ast_opt/ast_utilities/ProgramPrintVisitor.h"
#include "ast_opt/ast_parser/Parser.h"
#include "gtest/gtest.h"

TEST(ProgramPrintVisitor, printTree) {

  const char *programCode = R""""(
  public void main(secret int p, float q, bool x) {
    int scalar = 2;
    int vec = {3, 4, 9, 2, 1};
    scalar = 6 + 7;
    vec[7] = (10*5)+3;
    return 77;
    return 6;
    if (a > 5) {
      z = 5 + 7;
    } else {
      if (x > 6) {
        z = 6;
      }
      else if (y <  7) {
        y = 6;
      }
    }
  }
)"""";
  const char *resultCode =
      R""""({
  void main(secret int p, float q, bool x)
  {
    int scalar = 2;
    int vec = {3, 4, 9, 2, 1};
    scalar = (6 + 7);
    vec[7] = ((10 * 5) + 3);
    return 77;
    return 6;
    if((a > 5))
    {
      z = (5 + 7);
    }
    else
    {
      if((x > 6))
      {
        z = 6;
      }
      else
      {
        if((y < 7))
        {
          y = 6;
        }
      }
    }
  }
}
)"""";

  auto code = std::string(programCode);
  auto ast = Parser::parse(code);

  std::stringstream ss;
  ProgramPrintVisitor v(ss);
  ast->accept(v);

  std::cout << ss.str() << std::endl;

  EXPECT_EQ(ss.str(), resultCode);
}