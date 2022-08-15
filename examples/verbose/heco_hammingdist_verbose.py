import logging

# XXX: need to import everything to avoid translation failure for aliases.
#   This is a current limitation of the frontend.
from xdsl.frontend import *
from xdsl.frontend.dialects.builtin import *
from heco.frontend.dialects.fhe import *

p = FrontendProgram()

secret_f64 = SecretType[f64]

# TODO: @Alex, why did you use TensorType instead of BatchedSecretType here
#   in the original hammingdistance program?
#   The problem is Extract is only defined on BatchedSecretType, so the TensorType
#   versiondoes not work at the moment.
#   But BatchedSecretType has no shape information.

# TensorType version:
#shape: ArrayAttr[IndexType] = ArrayAttr[IntegerAttr[Literal[4]]]
#arg_type = TensorType[shape, secret_f64]

arg_type = BatchedSecretType[f64]

with CodeContext(p, log_level=logging.DEBUG):
    def encryptedHammingDistance(x: arg_type,
                                 y: arg_type) -> secret_f64:
        sum: SecretType[f64] = SecretAttr(FloatAttr(0.0))

        for idx in range(0, 4):
            sum = sum + (x[idx] - y[idx]) * (x[idx] - y[idx])

        return sum

p.compile()
