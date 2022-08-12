"module"() ({
  "func.func"() {"sym_name" = "encryptedHammingDistance", "function_type" = !fun<[!tensor<[4 : !index], !fhe.secret_type<!f64>>, !tensor<[4 : !index], !fhe.secret_type<!f64>>], []>, "sym_visibility" = "private"} ({
  ^0(%0 : !tensor<[4 : !index], !fhe.secret_type<!f64>>, %1 : !tensor<[4 : !index], !fhe.secret_type<!f64>>):
    "symref.declare"() {"sym_name" = "sum"} : () -> ()
    %2 = "arith.constant"() {"value" = !fhe.secret<0 : !i64, !fhe.secret_type<!f64>>} : () -> ()
    "symref.update"(%2) {"symbol" = @sum} : (!fhe.secret_type<!f64>) -> (!fhe.secret_type<!f64>)
    "affine.for"() {"lower_bound" = 0 : !index, "upper_bound" = 4 : !index, "step" = 1 : !index} ({
    ^1(%3 : !i32):
      %4 = "fhe.extract"(%0, %3) : (!tensor<[4 : !index], !fhe.secret_type<!f64>>, !i32) -> (!tensor<[4 : !index], !fhe.secret_type<!f64>>, !i32)
      %5 = "fhe.extract"(%1, %3) : (!tensor<[4 : !index], !fhe.secret_type<!f64>>, !i32) -> (!tensor<[4 : !index], !fhe.secret_type<!f64>>, !i32)
      %6 = "fhe.sub"(%4, %5) : (!fhe.secret_type<!f64>, !fhe.secret_type<!f64>) -> (!fhe.secret_type<!f64>, !fhe.secret_type<!f64>)
      %7 = "fhe.mul"(%6, %6) : (!fhe.secret_type<!f64>, !fhe.secret_type<!f64>) -> (!fhe.secret_type<!f64>, !fhe.secret_type<!f64>)
      %8 = "symref.fetch"() {"symbol" = @sum} : () -> ()
      %9 = "fhe.add"(%8, %7) : (!fhe.secret_type<!f64>, !fhe.secret_type<!f64>) -> (!fhe.secret_type<!f64>, !fhe.secret_type<!f64>)
      "symref.update"(%9) {"symbol" = @sum} : (!fhe.secret_type<!f64>) -> (!fhe.secret_type<!f64>)
    }) : () -> ()
    %10 = "symref.fetch"() {"symbol" = @sum} : () -> ()
    "func.return"(%10) : (!fhe.secret_type<!f64>) -> (!fhe.secret_type<!f64>)
  }) : () -> ()
}) : () -> ()
