#
# This code would normally be auto-generated.
#
# For the purpose of prototyping and testing, we manually create this file for now,
# until we have the auto-generation pipeline.
#
# TODO: current limitations of the manual prototype:
#   - Operations with :Attribute input parameters are only added to heco types for now.
#     The operation would actually also need to be added to all other Attributes to
#     support y + Secret(x) and not only Secret(x) + y.
#       --> Use overloads for this!

from dataclasses import dataclass
from typing import Generic, List, TypeVar, Union

from xdsl_frontend.dialects.builtin import IndexType, IntegerAttr, i32
from xdsl.ir import Attribute
import heco.dialects.fhe as orig_fhe

V = TypeVar("V", bound=Attribute, covariant=True)


################################################################################################
#                                             Types                                            #
################################################################################################

@dataclass
class SecretType(Generic[V], Attribute):
    """
    Secret.

    A `Type` for `Secret`s (high-level abstraction of `Ciphertext`s encrypting
    single scalars)

    name: "secret_type"

    Parameters:
        - plaintext_type: ParameterDef[Attribute]
    """

    # TODO: @Alex, wouldn't it make sense to make this more restrictive?
    #   Can Attribute only be of type V? (the plaintext type of SecretType)
    def __add__(self: 'SecretType[V]', other: Attribute) -> 'SecretType[V]':
        return Add(self, other)

    def __sub__(self: 'SecretType[V]', other: Attribute) -> 'SecretType[V]':
        return Sub(self, other)

    def __mul__(self: 'SecretType[V]', other: Attribute) -> 'SecretType[V]':
        return Mul(self, other)


@dataclass
class BatchedSecretType(Generic[V], Attribute):
    """
    Batched Secret.

    A Type for Batched `Secret`s (high-level abstraction of `Ciphertext`s
    encrypting vectors of values)

    name: "batched_secret_type"

    Parameters:
        - plaintext_type: ParameterDef[Attribute]
    """

    # TODO: @Alex, wouldn't it make sense to make this more restrictive?
    #   Can Attribute only be of type V? (the plaintext type of SecretType)
    def __add__(self: 'SecretType[V]', other: Attribute) -> 'SecretType[V]':
        return Add(self, other)

    def __sub__(self: 'SecretType[V]', other: Attribute) -> 'SecretType[V]':
        return Sub(self, other)

    def __mul__(self: 'SecretType[V]', other: Attribute) -> 'SecretType[V]':
        return Mul(self, other)

    def __getitem__(self, idx: IntegerAttr[IndexType]) -> SecretType[V]:
        return Extract(self, idx)

################################################################################################
#                                        Constraints                                           #
################################################################################################


# XXX: we need to find a way to auto-generate these. Maybe we could register this over AnyOf.
AnySecretType = Union[SecretType, BatchedSecretType]

_SecretTyp = TypeVar("_SecretTyp", bound=SecretType, covariant=True)


################################################################################################
#                                        Attributes                                            #
################################################################################################

@dataclass
class SecretAttr(SecretType[_SecretTyp]):
    """
    name: "secret"

    Parameters:
        - plaintext: ParameterDef[Attribute]
        - typ: ParameterDef[_SecretTyp]
    """
    plaintext: Attribute

    @staticmethod
    def _default_type_from_args(args: List[Attribute]) -> Attribute:
        return orig_fhe.SecretType([args[0].typ])


################################################################################################
#                                          Operation                                           #
################################################################################################

def Mul(*inputs: Attribute) -> AnySecretType:
    """
    Multiplication.

    name: "fhe.mul"

    Inputs:
        - inputs = VarOperandDef(AnyAttr())

    Returns:
        - output = ResultDef(AnySecretType)
    """
    # TODO: this is a hack to get the correct xDSL operation during the frontend translation.
    #   We should really find a nicer version, because here the return type is wrong.
    #   This only works because the frontend program is never evaluated, so for static
    #   type checking we only use the function signature.
    from heco.dialects.fhe import Mul
    return Mul  # type: ignore


def Add(*inputs: Attribute) -> AnySecretType:
    """
    Addition.

    name: "fhe.add"

    Inputs:
        - inputs = VarOperandDef(AnyAttr())

    Returns:
        - output = ResultDef(AnySecretType)
    """
    # TODO: this is a hack to get the correct xDSL operation during the frontend translation.
    #   We should really find a nicer version, because here the return type is wrong.
    #   This only works because the frontend program is never evaluated, so for static
    #   type checking we only use the function signature.
    from heco.dialects.fhe import Add
    return Add  # type: ignore


def Sub(*inputs: Attribute) -> AnySecretType:
    """
    Subtraction.

    name: "fhe.sub"

    Inputs:
        - inputs = VarOperandDef(AnyAttr())

    Returns:
        - output = ResultDef(AnySecretType)
    """
    # TODO: this is a hack to get the correct xDSL operation during the frontend translation.
    #   We should really find a nicer version, because here the return type is wrong.
    #   This only works because the frontend program is never evaluated, so for static
    #   type checking we only use the function signature.
    from heco.dialects.fhe import Sub
    return Sub  # type: ignore

# TODO: no SI32Attr exists yet in xDSL, so we use i32 for input1 instead.


def Rotate(input0: BatchedSecretType[V], input1: i32) -> BatchedSecretType[V]:
    """
    Rotate.

    name: "fhe.rotate"

    Inputs:
        - input0 = OperandDef(BatchedSecretType)
        - input1 = OperandDef(i32)

    Returns:
        - output = ResultDef(BatchedSecretType)
    """
    # TODO: this is a hack to get the correct xDSL operation during the frontend translation.
    #   We should really find a nicer version, because here the return type is wrong.
    #   This only works because the frontend program is never evaluated, so for static
    #   type checking we only use the function signature.
    from heco.dialects.fhe import Rotate
    return Rotate  # type: ignore


def Relineraization(inputs: AnySecretType) -> AnySecretType:
    """
    Relineraization.

    name: "fhe.relinearize"

    Inputs:
        - inputs = OperandDef(AnySecretType)

    Returns:
        - output = ResultDef(AnySecretType)
    """
    # TODO: this is a hack to get the correct xDSL operation during the frontend translation.
    #   We should really find a nicer version, because here the return type is wrong.
    #   This only works because the frontend program is never evaluated, so for static
    #   type checking we only use the function signature.
    from heco.dialects.fhe import Relineraization
    return Relineraization  # type: ignore


def Constant(inputs: Attribute) -> AnySecretType:
    """
    FHE constant.

    Cast a value into a `Secret` containing the same type.

    name: "fhe.constant"

    Inputs:
        - inputs = OperandDef(AnyAttr())

    Returns:
        - output = ResultDef(AnySecretType)
    """
    # TODO: this is a hack to get the correct xDSL operation during the frontend translation.
    #   We should really find a nicer version, because here the return type is wrong.
    #   This only works because the frontend program is never evaluated, so for static
    #   type checking we only use the function signature.
    from heco.dialects.fhe import Constant
    return Constant  # type: ignore


def Extract(input0: BatchedSecretType[V], input1: IntegerAttr[IndexType]) -> SecretType[V]:
    """
    Element extraction operation (actually executing this under FHE is highly inefficient).

    name: "fhe.extract"

    Inputs:
        - input0 = OperandDef(BatchedSecretType)
        - input1 = OperandDef(IndexType)

    Returns:
        - output = ResultDef(AnySecretType)
    """
    # TODO: this is a hack to get the correct xDSL operation during the frontend translation.
    #   We should really find a nicer version, because here the return type is wrong.
    #   This only works because the frontend program is never evaluated, so for static
    #   type checking we only use the function signature.
    from heco.dialects.fhe import Extract
    return Extract  # type: ignore


def Insert(input0: BatchedSecretType[V], input1: SecretAttr[V],
           input2: IntegerAttr[IndexType]) -> BatchedSecretType[V]:
    """
    Element insertion operation (actually executing this under FHE is highly inefficient).

    name: "fhe.insert"

    Inputs:
        - input0 = OperandDef(BatchedSecretType)
        - input1 = OperandDef(SecretAttr)
        - input2 = OperandDef(IndexType)

    Returns:
        - output = ResultDef(AnySecretType)
    """
    # TODO: this is a hack to get the correct xDSL operation during the frontend translation.
    #   We should really find a nicer version, because here the return type is wrong.
    #   This only works because the frontend program is never evaluated, so for static
    #   type checking we only use the function signature.
    from heco.dialects.fhe import Insert
    return Insert  # type: ignore

# XXX: CombineOp omitted for now


def Materialize(input: Attribute) -> Attribute:
    """
    No-op operation used to preserve consistency of type system during type conversion.

    name: "fhe.materialize"

    Inputs:
        - input = OperandDef(AnyAttr())

    Returns:
        - output = ResultDef(AnyAttr())
    """
    # TODO: this is a hack to get the correct xDSL operation during the frontend translation.
    #   We should really find a nicer version, because here the return type is wrong.
    #   This only works because the frontend program is never evaluated, so for static
    #   type checking we only use the function signature.
    from heco.dialects.fhe import Materialize
    return Materialize  # type: ignore
