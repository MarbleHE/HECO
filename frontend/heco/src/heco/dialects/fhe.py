from dataclasses import dataclass

from xdsl.irdl import *
from xdsl.ir import *
from xdsl.dialects.builtin import *
from xdsl.printer import Printer


@dataclass
class FHE:
    """
    FHE Dialect.

    This dialect represents a common abstraction for FHE operations.
    """
    ctx: MLContext

    def __post_init__(self):
        self.ctx.register_attr(SecretType)
        self.ctx.register_attr(BatchedSecretType)

        self.ctx.register_op(Mul)
        self.ctx.register_op(Add)
        self.ctx.register_op(Sub)
        self.ctx.register_op(Rotate)
        self.ctx.register_op(Relineraization)
        self.ctx.register_op(Constant)
        self.ctx.register_op(Extract)
        self.ctx.register_op(Insert)
        self.ctx.register_op(Materialize)


#===----------------------------------------------------------------------===#
# FHE print helpers.
#===----------------------------------------------------------------------===#

class PrettyPrinter:

    def __init__(self, printer: Printer) -> None:
        self.printer = printer

    def _print_type(self, value: SSAValue) -> None:
        self.printer.print(" : ")
        self.printer.print_attribute(value.typ)

    def _print_types(self, values: List[SSAValue]) -> None:
        for value in values[:-1]:
            self.printer.print_attribute(value.typ)
            self.printer.print(", ")
        self.printer.print_attribute(values[-1].typ)

    def _print_signature(self, operation: Operation) -> None:
        self.printer.print(" : (")
        self._print_types(operation.operands)
        self.printer.print(") -> ")
        self._print_types(operation.results)

    def _print_operands(self, operation: Operation) -> None:
        operands = operation.operands
        self.printer.print("(")
        for op in operands[:-1]:
            self.printer.print(op, ", ")
        self.printer.print(operands[-1], ")")

    def _print_attributes(self, operation: Operation) -> None:
        self.printer.print(" ", operation.attributes)

    def print_binop(self, operation: Operation) -> None:
        self._print_operands(operation)
        self._print_attributes(operation)
        self._print_signature(operation)

    def _print_vector(self, vector: ParametrizedAttribute, idx: IndexType) -> None:
        self.printer.print(vector, "[", idx, "]")

    def print_vector_extract(self, operation: Operation) -> None:
        assert(len(operation.operands) == 2 and len(operation.results) == 1)
        self._print_vector(*operation.operands)
        self._print_attributes(operation)
        self._print_type(operation.results[0])

    def print_vector_insert(self, operation: Operation) -> None:
        assert(len(operation.operands) == 3 and len(operation.results) == 1)
        vector, elem, idx = operation.operands
        self.printer.print(elem, " into ")
        self._print_vector(vector, idx)
        self._print_attributes(operation)
        self._print_type(operation.results[0])

    def print_constant(self, operation: Operation) -> None:
        assert(len(operation.operands) == 1)
        self.printer.print_attribute(operation)
        self.printer.print(" ", operation.operands[0])

    def print_rotate(self, operation: Operation) -> None:
        # `(` $x `)` `by` $i attr-dict `:` type($x)
        assert(len(operation.operands) == 1)
        self._print_operands(operation)
        self.printer.print(" by ", operation.operands[0], " ")
        self._print_attributes(operation)
        self._print_type(operation.operands[0])

#===----------------------------------------------------------------------===#
# Other helpers.
#===----------------------------------------------------------------------===#


def _derive_result_type(value: Union[Operation, SSAValue]) -> Attribute:
    if isinstance(value, Operation):
        if len(value.results) != 1:
            raise Exception(
                f"Expected exactly one result, got {len(value.results)}")

        result = value.results[0]
        if not hasattr(result, "typ"):
            raise Exception(
                f"Attribute {result} does not have operand 'typ' storing its type.")

        return result.typ
    elif isinstance(value, SSAValue):
        return value.typ
    else:
        raise Exception(
            f"Cannot derive value from type {type(value)}; {value}.")


def _unwrap_type(attr: Attribute, recursive=True) -> Attribute:
    if isinstance(attr, SecretType):
        return attr.plaintext_type
    elif isinstance(attr, BatchedSecretType):
        return attr.plaintext_type
    elif isinstance(attr, TensorType):
        if recursive:
            return _unwrap_type(attr.element_type)
        else:
            return attr.element_type
    else:
        raise Exception(
            f"Cannot access wrapped type of {attr} (of type {type(attr)}).")

#===----------------------------------------------------------------------===#
# FHE type definitions.
#===----------------------------------------------------------------------===#


@irdl_attr_definition
class SecretType(ParametrizedAttribute):
    """
    Secret.

    A `Type` for `Secrets` (high-level abstraction of `Ciphertexts` encrypting
    single scalars)
    """
    name: str = "fhe.secret_type"
    plaintext_type: ParameterDef[Attribute]

    def get_carried_type(self) -> Attribute:
        return self.plaintext_type

    @staticmethod
    @builder
    def from_plaintext_type(typ: Attribute) -> 'SecretType':
        return SecretType([typ])

    # TODO: custom printing?
    # "`<` $plaintextType `>`";


@irdl_attr_definition
class BatchedSecretType(ParametrizedAttribute):
    """
    Batched Secret.

    A Type for Batched `Secrets` (high-level abstraction of `Ciphertexts`
    encrypting vectors of values)
    """

    name: str = "fhe.batched_secret_type"
    plaintext_type: ParameterDef[Attribute]

    # TODO: how can we add inlined C++ code?
    # let extraClassDeclaration = [{
    #         SecretType getCorrespondingSecretType() const;
    #         /// Sets size = -1
    #         static BatchedSecretType get(::mlir::MLIRContext *context, ::mlir::Type plaintextType);
    # }];

    @staticmethod
    @builder
    def from_plaintext_type(typ: Attribute) -> 'BatchedSecretType':
        return BatchedSecretType([typ])

    # TODO: custom printing
    # "`<` $size `x` $plaintextType `>`"


AnySecretType: AttrConstraint = AnyOf([SecretType, BatchedSecretType])

_SecretTyp = TypeVar("_SecretTyp", bound=SecretType, covariant=True)


@irdl_attr_definition
class SecretAttr(Generic[_SecretTyp], ParametrizedAttribute):
    name = "fhe.secret"
    plaintext: ParameterDef[Attribute]
    typ: ParameterDef[_SecretTyp]

    @staticmethod
    @builder
    def from_plaintext_and_type(plaintext: Attribute, typ: _SecretTyp) -> 'SecretAttr[_SecretTyp]':
        return SecretAttr([plaintext, typ])


#===----------------------------------------------------------------------===#
# FHE Operations
#===----------------------------------------------------------------------===#


@irdl_op_definition
class Mul(Operation):
    """Multiplication."""

    name: str = "fhe.mul"

    # TODO: how can we specify traits?
    # Commutative, DeclareOpInterfaceMethods<InferTypeOpInterface>, NoSideEffect

    arguments = VarOperandDef(AnyAttr())

    output = ResultDef(AnySecretType)

    # TODO: what is "hasFolder", do we need it, and how can we specify it?
    # hasFolder = 1

    @staticmethod
    def get(*operands: Operation | SSAValue,
            result_type: AnySecretType) -> 'Mul':
        return Mul.build(operands=[list(operands)], result_types=[result_type])

    def print(self, printer: Printer) -> None:
        pp = PrettyPrinter(printer)
        pp.print_binop(self)


@irdl_op_definition
class Add(Operation):
    """Addition."""

    name: str = "fhe.add"

    # TODO: how can we specify traits?
    # Commutative, DeclareOpInterfaceMethods<InferTypeOpInterface>, NoSideEffect

    arguments = VarOperandDef(AnyAttr())

    output = ResultDef(AnySecretType)

    # TODO: what is "hasFolder", do we need it, and how can we specify it?
    # hasFolder = 1

    @staticmethod
    def get(*operands: Operation | SSAValue, result_type: AnySecretType) -> 'Add':
        return Add.build(operands=[list(operands)], result_types=[result_type])

    def print(self, printer: Printer) -> None:
        pp = PrettyPrinter(printer)
        pp.print_binop(self)


@irdl_op_definition
class Sub(Operation):
    """Subtraction."""

    name: str = "fhe.sub"

    # TODO: how can we specify traits?
    # DeclareOpInterfaceMethods<InferTypeOpInterface>, NoSideEffect

    arguments = VarOperandDef(AnyAttr())

    output = ResultDef(AnySecretType)

    # TODO: what is "hasFolder", do we need it, and how can we specify it?
    # hasFolder = 1

    @staticmethod
    def get(*operands: Operation | SSAValue,
            result_type: AnySecretType) -> 'Sub':
        return Sub.build(operands=[list(operands)], result_types=[result_type])

    def print(self, printer: Printer) -> None:
        pp = PrettyPrinter(printer)
        pp.print_binop(self)


@irdl_op_definition
class Rotate(Operation):
    """Rotate."""

    name: str = "fhe.rotate"

    # TODO: how can we specify traits?
    # AllTypesMatch<["x","output"]>, NoSideEffect

    input0 = OperandDef(BatchedSecretType)
    # TODO: no SI32Attr exists yet in xDSL. We may want to add a `_verify` check
    #   that this value is positive.
    input1 = OperandDef(i32)

    output = ResultDef(BatchedSecretType)

    # TODO: what is "hasFolder", do we need it, and how can we specify it?
    # hasFolder = 1

    @staticmethod
    def get(vector: Union[Operation, SSAValue],
            shift: Union[Operation, SSAValue]) -> 'Relineraization':
        result_type = _derive_result_type(vector)
        return Relineraization.build(operands=[vector, shift],
                                     result_types=[result_type])

    def print(self, printer: Printer) -> None:
        printer.print(" ")  # XXX: we cannot remove the operation name
        pp = PrettyPrinter(printer)
        pp.print_rotate(self)


@irdl_op_definition
class Relineraization(Operation):
    """Relinearization."""
    name: str = "fhe.relinearize"

    # TODO: how can we specify traits?
    # AllTypesMatch<["x", "output"], NoSideEffect

    inputs = OperandDef(AnySecretType)
    output = ResultDef(AnySecretType)

    @staticmethod
    def get(input: Union[Operation, SSAValue],
            typ: AnySecretType) -> 'Relineraization':
        return Relineraization.build(operands=[input], result_types=[typ])


@irdl_op_definition
class Constant(Operation):
    """
    FHE constant.

    Cast a value into a secret containing the same type.
    """
    # TODO: what happens in TableGen when you use the same variable twice?
    # Constat defines summary twice.
    name: str = "fhe.constant"

    # TODO: how can we specify traits?
    # ConstantLike, InferTypeOpInterface,
    # DeclareOpInterfaceMethods<InferTypeOpInterface>,
    # DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"],
    # NoSideEffect

    inputs = OperandDef(AnyAttr())

    output = ResultDef(AnySecretType)

    @staticmethod
    def get(value: Union[Operation, SSAValue],
            typ: AnySecretType) -> 'Constant':
        return Constant.build(operands=[value], result_types=[typ])

    def print(self, printer: Printer) -> None:
        printer.print(" ")  # XXX: we cannot remove the operation name
        pp = PrettyPrinter(printer)
        pp.print_constant(self)


@irdl_op_definition
class Extract(Operation):
    """
    Element extraction operation (actually executing this under FHE is highly inefficient).
    """

    # TODO: what happens in TableGen when you use the same variable twice?
    # Constant defines summary twice.

    name: str = "fhe.extract"

    # TODO: how can we specify traits?
    # TypesMatchWith<"result type matches element type of tensor",
    #                "vector", "result",
    #                "$_self.cast<BatchedSecretType>().getCorrespondingSecretType()">
    #                NoSideEffect

    input0 = OperandDef(BatchedSecretType)
    input1 = OperandDef(IntegerAttr)

    output = ResultDef(SecretType)

    @staticmethod
    def get(vector: Union[Operation, SSAValue],
            idx: Union[Operation, SSAValue]) -> 'Extract':
        result_type = SecretType.from_plaintext_type(
            _unwrap_type(_derive_result_type(vector)))
        return Extract.build(operands=[vector, idx], result_types=[result_type])

    def print(self, printer: Printer) -> None:
        printer.print(" ")  # XXX: we cannot remove the operation name
        pp = PrettyPrinter(printer)
        pp.print_vector_extract(self)


@irdl_op_definition
class Insert(Operation):
    """
    Element insertion operation (actually executing this under FHE is highly inefficient).
    """

    name: str = "fhe.insert"

    # TODO: how can we specify traits?
    # AllTypesMatch<["dest", "result"]>,
    # TypesMatchWith<"scalar type matches element type of dest",
    #                "dest", "scalar",
    #                "$_self.cast<BatchedSecretType>().getCorrespondingSecretType()">
    # NoSideEffect

    input0 = OperandDef(BatchedSecretType)
    input1 = OperandDef(SecretAttr)
    input2 = OperandDef(IntegerAttr)

    output = ResultDef(BatchedSecretType)

    # TODO: what is "hasCanonicalizeMethod", do we need it, and how can we specify it?
    # hasCanonicalizeMethod = 1

    @staticmethod
    def get(vector: Union[Operation, SSAValue],
            elem: Union[Operation, SSAValue],
            idx: Union[Operation, SSAValue]) -> 'Insert':

        result_type = BatchedSecretType.from_plaintext_type(
            _unwrap_type(_derive_result_type(elem)))
        return Insert.build(operands=[vector, elem, idx],
                            result_types=[result_type])

    def print(self, printer: Printer) -> None:
        printer.print(" ")  # XXX: we cannot remove the operation name
        pp = PrettyPrinter(printer)
        pp.print_vector_insert(self)


# XXX: CombineOp omitted for now


@irdl_op_definition
class Materialize(Operation):
    """
    No-op operation used to preserve consistency of type system during type conversion.
    """

    name: str = "fhe.materialize"

    # TODO: how can we specify traits?
    # TypesMatchWith<"result type matches element type of tensor",
    #                "vector", "result",
    #                "$_self.cast<BatchedSecretType>().getCorrespondingSecretType()">
    #                NoSideEffect

    input = OperandDef(AnyAttr())

    output = ResultDef(AnyAttr())

    # TODO: what is "hasCanonicalizeMethod" and "hasFolder" and do we need them?
    # hasFolder = 1
    # hasCanonicalizeMethod = 1

    @staticmethod
    def get(input: Union[Operation, SSAValue], result_type: Attribute) -> 'Materialize':
        return Materialize.build(operands=[input],
                                 result_types=[result_type])
