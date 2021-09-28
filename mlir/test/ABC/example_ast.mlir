// RUN: abc-opt %s | abc-opt | FileCheck %s

"abc.function"() ({
   //"abc.function_parameter"() {type = "secret int", name = "y"} : () -> ()
}, {
    //"abc.variable_declaration"() ({
    //    "abc.literal_int"() {value = "17"} : () -> ()
    // }) {type = "int", name = "x"}: () -> ()

}) {name = "foo", return_type = "void"}: () -> ()