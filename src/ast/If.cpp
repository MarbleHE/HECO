#include "../../include/ast/If.h"

If::If(AbstractExpr *condition, AbstractStatement *thenBranch, AbstractStatement *elseBranch) : condition(condition),
                                                                                                thenBranch(thenBranch),
                                                                                                elseBranch(
                                                                                                        elseBranch) {}
