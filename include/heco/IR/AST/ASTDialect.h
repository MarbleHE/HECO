#ifndef HECO_IR_AST_ASTDIALECT_H
#define HECO_IR_AST_ASTDIALECT_H

#include <mlir/IR/PatternMatch.h>
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir
{
    namespace OpTrait
    {
        template <typename ConcreteType>
        class isAbcExpression : public mlir::OpTrait::TraitBase<ConcreteType, isAbcExpression>
        {};

        template <typename ConcreteType>
        class isAbcStatement : public mlir::OpTrait::TraitBase<ConcreteType, isAbcStatement>
        {};

        template <typename ConcreteType>
        class isAbcTarget : public mlir::OpTrait::TraitBase<ConcreteType, isAbcTarget>
        {};
    } // namespace OpTrait
} // namespace mlir

#include "heco/IR/AST/ASTDialect.h.inc"

bool containsExactlyOneExpressionNode(mlir::Region &region);

bool containsExactlyOneTargetNode(mlir::Region &region);

bool containsExactlyOneStatementNode(mlir::Region &region);

#define GET_OP_CLASSES
#include "heco/IR/AST/AST.h.inc"

#define GET_TYPEDEF_CLASSES
#include "heco/IR/AST/ASTTypes.h.inc"

#endif // HECO_IR_AST_ASTDIALECT_H
