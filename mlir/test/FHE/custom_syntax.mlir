%0 = fhe.load_ctxt {file = "foo.ctxt", parms = "foo.parms"} : tensor<4096x2x2xi64>
%1 = fhe.load_ctxt {file = "boo.ctxt", parms = "boo.parms"} :  tensor<4096x2x2xi64>
%2 = fhe.multiply(%0,%1) {parms= "goo.parms"} : (tensor<4096x2x2xi64>, tensor<4096x2x2xi64>) -> tensor<4096x2x3xi64>
fhe.sink(%2) : (tensor<4096x2x3xi64>)