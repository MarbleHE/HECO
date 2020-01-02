#ifndef MASTER_THESIS_CODE_ASTTESTINGGENERATOR_H
#define MASTER_THESIS_CODE_ASTTESTINGGENERATOR_H

#include <Ast.h>

class AstTestingGenerator {
 private:
  static void genAstRewritingOne(Ast &ast);

  static void genAstRewritingTwo(Ast &ast);

  static void genAstRewritingThree(Ast &ast);

  static void genAstRewritingFour(Ast &ast);

  static void genAstRewritingFive(Ast &ast);

  static void genAstRewritingSix(Ast &ast);

 public:
  static void generateAst(int id, Ast &ast);
};

#endif //MASTER_THESIS_CODE_ASTTESTINGGENERATOR_H
