// RUN: abc-opt %s -mlir-print-op-on-diagnostic -mlir-print-stacktrace-on-diagnostic
//| abc-opt
//| FileCheck %s

"abc.function"() ({
   "abc.function_parameter"() {type = "secret int", name = "y"} : () -> ()
}, {
    "abc.block" () ({
        "abc.variable_declaration"() ({
           "abc.literal_int"() {value = 17} : () -> ()
        }) {type = "int", name = "x"}: () -> ()
        "abc.assignment" () ({
            "abc.variable" () {name = "foo"} : () -> ()
         }, {
            "abc.literal_int" () {value = 5} : () -> ()
         }) : () -> ()
         "abc.for" () ({
            // initializer
            "abc.block" () ({
                // ...
            }) : () -> ()
         },{
            // condition
            "abc.literal_bool" () {value = true} : () -> ()
         },{
            // update
            "abc.block" () ({
                // ...
            }) : () -> ()
        },{
            // body
            "abc.block" () ({
                // if with both branches
                "abc.if" () ({
                    // condition
                    "abc.literal_bool" () {value = true} : () -> ()
                }, {
                    // then
                    "abc.block" () ({
                        // ...
                    }) : () -> ()
                }, {
                     // else
                    "abc.block" () ({
                        // ...
                    }) : () -> ()
                }) : () -> ()
                // if with only then branch
                "abc.if" () ({
                    // condition
                    "abc.literal_bool" () {value = true} : () -> ()
                }, {
                    // then
                    "abc.block" () ({
                        // ...
                    }) : () -> ()
                }) : () -> ()
            }) : () -> ()
        }) : () -> ()
    }) : () -> ()

}) {name = "foo", return_type = "void"}: () -> ()