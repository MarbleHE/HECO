#ifndef AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_VECTORIZER_H_
#define AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_VECTORIZER_H_
#include "ast_opt/visitor/ScopedVisitor.h"
class SpecialVectorizer;

typedef Visitor<SpecialVectorizer> Vectorizer;

class SpecialVectorizer : public ScopedVisitor {

};
#endif //AST_OPTIMIZER_INCLUDE_AST_OPT_VISITOR_VECTORIZER_H_
