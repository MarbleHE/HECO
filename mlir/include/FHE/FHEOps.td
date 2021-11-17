#ifndef AST_OPTIMIZER_MLIR_FHE_FHEOPS_H_
#define AST_OPTIMIZER_MLIR_FHE_FHEOPS_H_

include "FHEDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def ABC_LoadCtxtOp : FHE_Op<"load_ctxt", []> {
  let summary = "Loads ciphertext from file into a register.";

  let arguments = (ins
    StrAttr: $file,
    StrAttr: $parms
  );

  let results = (outs
    TensorOf<[I64]>:$output
  );

  let assemblyFormat = [{
    attr-dict `:` type($output)
  }];
}

def ABC_MultiplyOp : FHE_Op<"multiply", []> {
  let summary = "Multiplication.";

  let arguments = (ins
    TensorOf<[I64]>:$x,
    TensorOf<[I64]>:$y,
    StrAttr: $parms
  );

  let results = (outs
    TensorOf<[I64]>:$output
  );

  let assemblyFormat = [{
    `(` $x `,` $y `)`  attr-dict `:`  `(` type($x) `,` type($y) `)`  `->` type($output)
  }];

}

def ABC_SinkOp : FHE_Op<"sink", [Terminator]> {
  let summary = "An operation that requires the operand to be evaluated (e.g. decrypt, serialize, etc).";

  let arguments = (ins
    TensorOf<[I64]>:$x
  );

  let assemblyFormat = [{
    `(` $x`)`  attr-dict `:`  `(` type($x) `)`
  }];

}

#endif // AST_OPTIMIZER_MLIR_FHE_FHEOPS_H_
