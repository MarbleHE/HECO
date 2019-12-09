#ifndef MASTER_THESIS_CODE_FUNCTION_H
#define MASTER_THESIS_CODE_FUNCTION_H


#include <string>
#include <vector>
#include "AbstractStatement.h"
#include "FunctionParameter.h"
#include "VarDecl.h"

class Function : public AbstractStatement {
public:
    std::string name;
    std::vector<FunctionParameter> params;


    std::vector<std::unique_ptr<AbstractStatement>> body;

    /// Copy constructor
    /// \param func The function to be copied.
    Function(const Function &func);

    Function(std::string name, std::vector<std::unique_ptr<AbstractStatement>> bodyStatements);

    void addParameter(const FunctionParameter &param);

};


#endif //MASTER_THESIS_CODE_FUNCTION_H
