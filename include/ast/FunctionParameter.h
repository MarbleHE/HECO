#ifndef MASTER_THESIS_CODE_FUNCTIONPARAMETER_H
#define MASTER_THESIS_CODE_FUNCTIONPARAMETER_H


#include "Variable.h"

class FunctionParameter : public Variable {
private:
    std::string datatype;

public:
    FunctionParameter(const std::string &identifier, std::string datatype);

    json toJson() const override;

    virtual void accept(Visitor &v) override;

    std::string getNodeName() const override;

    const std::string &getDatatype() const;
};

/// Defines the JSON representation to be used for vector<FunctionParameter> objects.
void to_json(json &j, const FunctionParameter &funcParam);


#endif //MASTER_THESIS_CODE_FUNCTIONPARAMETER_H
