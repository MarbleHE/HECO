from os.path import abspath, dirname

import heco.dialects.fhe as fhe

from xdsl.dialects.affine import For
from xdsl.dialects.arith import Constant
from xdsl.dialects.builtin import IndexType, IntegerAttr, ModuleOp, f64, i32
from xdsl.dialects.symref import Declare, Fetch, Update
from xdsl.dialects.func import FuncOp, Return
from xdsl.ir import Block, Region
from xdsl.printer import Printer

root = abspath(dirname(__file__))
printer = Printer(stream=open(f"{root}/out.mlir", mode="w+"),
                  print_generic_format=True, target=Printer.Target.MLIR,
                  print_operand_types=Printer.TypeLocation.AFTER,
                  print_result_types=Printer.TypeLocation.AFTER)

secret_f64 = fhe.SecretType.from_plaintext_type(f64)
tensor_4x_secret_f64 = fhe.TensorType.from_type_and_list(secret_f64, [4])

# Block takes two f64 arguments: x, y
block = Block.from_arg_types([tensor_4x_secret_f64, tensor_4x_secret_f64])

# int sum = 0;
sum_declare = Declare.get("sum")
sum_val = Constant.from_attr(
    fhe.SecretAttr.from_plaintext_and_type(
        IntegerAttr.from_int_and_width(0, 64),
        secret_f64
    ),
    secret_f64
)
sum_update = Update.get(
    "sum",
    sum_val
)
block.add_ops([sum_declare, sum_val, sum_update])

# TODO: Verify that the loop index is actually used as input for the block in the for loop body.
for_loop_body_block = Block.from_arg_types([i32])

# x[i]
xi = fhe.Extract.get(block.args[0], for_loop_body_block.args[0])

# y[i]
yi = fhe.Extract.get(block.args[1], for_loop_body_block.args[0])

# x[i] - y[i]
sub_xi_yi = fhe.Sub.get(xi, yi, result_type=secret_f64)

# (x[i] - y[i]) * (x[i] - y[i])
square_sub_xi_yi = fhe.Mul.get(sub_xi_yi, sub_xi_yi, result_type=secret_f64)

# sum + (x[i] - y[i]) * (x[i] - y[i])
sum_fetch = Fetch.get("sum", secret_f64)
sum_xi_yi = fhe.Add.get(sum_fetch, square_sub_xi_yi, result_type=secret_f64)

# sum = sum + (x[i] - y[i]) * (x[i] - y[i])
sum_update = Update.get(
    "sum",
    sum_xi_yi
)

for_loop_body_block.add_ops([xi, yi, sub_xi_yi, square_sub_xi_yi, sum_fetch, sum_xi_yi, sum_update])
for_loop_body_region = Region.from_block_list([for_loop_body_block])

for_loop = For.from_region(
    [],
    0,
    4,
    for_loop_body_region
)
block.add_op(for_loop)

sum_val = Fetch.get("sum", secret_f64)
return_op = Return.get(sum_val)
block.add_ops([sum_val, return_op])

fn_body = Region.from_block_list([block])
func = FuncOp.from_region("encryptedHammingDistance",
                          [tensor_4x_secret_f64, tensor_4x_secret_f64], [], fn_body)

module = ModuleOp.from_region_or_ops([func])
printer.print_op(module)
