#ifndef AST_OPTIMIZER_MLIR_FHE_FHEOPS_H_
#define AST_OPTIMIZER_MLIR_FHE_FHEOPS_H_

include "FHE/FHEDialect.td"
include "mlir/Dialect/Shape/IR/ShapeBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"

def FHE_LoadCtxtOp : FHE_Op<"load_ctxt", []> {
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

def FHE_MultiplyOp : FHE_Op<"multiply", []> {
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

def FHE_SinkOp : FHE_Op<"sink", [Terminator, ReturnLike]> {
  let summary = "An operation that requires the operand to be evaluated (e.g. decrypt, serialize, etc).";

  let arguments = (ins
    TensorOf<[I64]>:$x
  );

  let assemblyFormat = [{
    `(` $x`)`  attr-dict `:`  `(` type($x) `)`
  }];

}

// TODO: Move everything below to its own dialect

def FHE_GetPolyOp : FHE_Op<"to_poly", []> {
    let summary = "Gets the i-th polynomial from ctxt x.";

    let arguments = (ins
        TensorOf<[I64]>:$x,
        I64Attr:$i
    );

    let results = (outs
        TensorOf<[I64]>:$output
    );

    let assemblyFormat = [{
        `(` $x `)`  attr-dict `:`  `(` type($x) `)` `->` type($output)
    }];
}

def FHE_MakeCtxtOp : FHE_Op<"to_ctxt", []> {
    let summary = "Builds ctxt x from polyonmials";

    //TODO: Make this variadic
    let arguments = (ins
        TensorOf<[I64]>:$p1,
        TensorOf<[I64]>:$p2,
        TensorOf<[I64]>:$p3
    );

    let results = (outs
        TensorOf<[I64]>:$output
    );

    let assemblyFormat = [{
        `(` $p1 `,` $p2 `,` $p3 `)`  attr-dict `:`  `(` type($p1) `,` type($p2) `,` type($p3) `)` `->` type($output)
    }];
}



def FHE_PolyAddOp : FHE_Op<"poly_add", [NoSideEffect, Commutative]> {
  let summary = "Polynomial Addition.";

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

def FHE_PolyMulOp : FHE_Op<"poly_mul", [NoSideEffect, Commutative]> {
  let summary = "Polynomial Addition.";

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

def FHE_PolyMulAccOp : FHE_Op<"poly_macc", [NoSideEffect]> {
  let summary = "Polynomial Addition.";

  let arguments = (ins
    TensorOf<[I64]>:$x,
    TensorOf<[I64]>:$y,
    TensorOf<[I64]>:$acc,
    StrAttr: $parms
  );

  let results = (outs
    TensorOf<[I64]>:$output
  );

  let assemblyFormat = [{
    `(` $x `,` $y `,` $acc `)`  attr-dict `:`  `(` type($x) `,` type($y) `,` type($acc) `)`  `->` type($output)
  }];

}
#endif // AST_OPTIMIZER_MLIR_FHE_FHEOPS_H_
