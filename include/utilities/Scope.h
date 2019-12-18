#ifndef MASTER_THESIS_CODE_SCOPE_H
#define MASTER_THESIS_CODE_SCOPE_H


#include <map>
#include <queue>
#include "../ast/AbstractStatement.h"


class Scope {

private:
    std::map<std::string, Scope *> innerScopes;
    Scope *outerScope;
    std::string scopeIdentifier;
    std::vector<AbstractStatement *> scopeStatements;

public:
    Scope(std::string scopeIdentifier, Scope *outerScope);

    [[nodiscard]] Scope *getOuterScope() const;

    [[nodiscard]] const std::string &getScopeIdentifier() const;

    [[nodiscard]] const std::vector<AbstractStatement *> &getScopeStatements() const;

    Scope *getOrCreateInnerScope(const std::string &identifier);

    Scope *findInnerScope(const std::string &identifier);

    void addStatement(AbstractStatement *absStatement);

    AbstractStatement *getNthLastStatement(int n);
};


#endif //MASTER_THESIS_CODE_SCOPE_H
