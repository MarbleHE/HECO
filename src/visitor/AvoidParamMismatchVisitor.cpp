#include "ast_opt/ast/AbstractNode.h"
#include "ast_opt/ast/Literal.h"
#include "ast_opt/visitor/IdentifyNoisySubtreeVisitor.h"
#include "ast_opt/visitor/AvoidParamMismatchVisitor.h"
#include <utility>
#include "ast_opt/ast/AbstractExpression.h"

SpecialAvoidParamMismatchVisitor::SpecialAvoidParamMismatchVisitor(std::unordered_map<std::string, std::vector<seal::Modulus>> coeffmodulusmap)
    : coeffmodulusmap(std::move(coeffmodulusmap)) {}

void SpecialAvoidParamMismatchVisitor::visit(BinaryExpression &elem) {

};

