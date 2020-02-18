#ifndef AST_OPTIMIZER_INCLUDE_FUNCTION_H
#define AST_OPTIMIZER_INCLUDE_FUNCTION_H

#include <string>
#include <vector>
#include "AbstractStatement.h"
#include "FunctionParameter.h"
#include "VarDecl.h"

class Function : public AbstractStatement {
private:
    std::string name;
    std::vector<FunctionParameter *> params;
    std::vector<AbstractStatement *> body;
public:
    Function() = default;

    Function *clone(bool keepOriginalUniqueNodeId) override;

    [[nodiscard]] const std::string &getName() const;

    [[nodiscard]] const std::vector<FunctionParameter *> &getParams() const;

    [[nodiscard]] const std::vector<AbstractStatement *> &getBody() const;

    Function(std::string name, std::vector<AbstractStatement *> bodyStatements);

    Function(std::string functionName, std::vector<FunctionParameter *> functionParameters,
             std::vector<AbstractStatement *> functionStatements);

    explicit Function(std::string name);

    void addParameter(FunctionParameter *param);

    void addStatement(AbstractStatement *pDecl);

    [[nodiscard]] json toJson() const override;

    void accept(Visitor &v) override;

    [[nodiscard]] std::string getNodeName() const override;

    void setParams(std::vector<FunctionParameter *> paramsVec);
};

/// Defines the JSON representation to be used for vector<Function> objects.
void to_json(json &j, const Function &func);

#endif //AST_OPTIMIZER_INCLUDE_FUNCTION_H
