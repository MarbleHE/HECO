#ifndef MASTER_THESIS_CODE_ASTTESTINGGENERATOR_H
#define MASTER_THESIS_CODE_ASTTESTINGGENERATOR_H


#include <Ast.h>

class AstTestingGenerator {
private:
    static void genAstRewritingOne(Ast &ast);

    static void genAstRewritingTwo(Ast &ast);

public:
    static void getRewritingAst(int id, Ast &ast);
};


#endif //MASTER_THESIS_CODE_ASTTESTINGGENERATOR_H
