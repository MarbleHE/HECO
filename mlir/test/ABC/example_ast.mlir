// RUN: abc-opt %s -mlir-print-op-on-diagnostic -mlir-print-stacktrace-on-diagnostic
//| abc-opt
//| FileCheck %s

"abc.function"() ({
   "abc.function_parameter"() {type = !abc.int, name = "y"} : () -> ()
}, {
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
        "abc.variable_declaration"() ({
           "abc.literal_int"() {value = 17} : () -> ()
        }) {type = i32, name = "x"}: () -> ()
        "abc.assignment" () ({
            "abc.variable" () {name = "foo"} : () -> ()
         }, {
            "abc.literal_int" () {value = 5} : () -> ()
         }) : () -> ()
         "abc.for" () ({
            // initializer
            "abc.block" () ({
                "abc.variable_declaration"() ({
                    "abc.literal_int"() {value = 7} : () -> ()
                }) {type = i32, name = "i"}: () -> ()
            }) : () -> ()
         },{
            // condition
            "abc.literal_bool" () {value = true} : () -> ()
         },{
            // update
            "abc.block" () ({
                "abc.assignment" () ({
                    "abc.variable" () {name = "i"} : () -> ()
                }, {
                    "abc.literal_int" () {value = 5} : () -> ()
                }) : () -> ()
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
        "abc.return"() ({
            "abc.literal_int" () {value = 5} : () -> ()
        }) : () -> ()
    }) : () -> ()

}) {name = "foo", return_type = i1}: () -> ()