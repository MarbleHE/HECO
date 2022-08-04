module {
  func private @trace() -> !emitc.opaque<"seal::Ciphertext"> {
    %0 = "emitc.constant"() {value = #emitc.opaque<"\22p_1.parms\22">} : () -> !emitc.opaque<"std::string">
    %1 = "emitc.constant"() {value = #emitc.opaque<"\22encrpytion_3.ctxt\22">} : () -> !emitc.opaque<"std::string">
    %2 = "emitc.constant"() {value = #emitc.opaque<"\22encrpytion_2.ctxt\22">} : () -> !emitc.opaque<"std::string">
    %3 = "emitc.constant"() {value = #emitc.opaque<"\22p_0.parms\22">} : () -> !emitc.opaque<"std::string">
    %4 = "emitc.constant"() {value = #emitc.opaque<"\22relin_keys_1.rk\22">} : () -> !emitc.opaque<"std::string">
    %5 = "emitc.constant"() {value = #emitc.opaque<"\22public_key_0.pk\22">} : () -> !emitc.opaque<"std::string">
    %6 = emitc.call "evaluator_load_public_key"(%5, %3) : (!emitc.opaque<"std::string">, !emitc.opaque<"std::string">) -> !emitc.opaque<"seal::PublicKey">
    %7 = emitc.call "evaluator_load_relin_keys"(%4, %3) : (!emitc.opaque<"std::string">, !emitc.opaque<"std::string">) -> !emitc.opaque<"seal::RelinKeys">
    %8 = emitc.call "evaluator_load_ctxt"(%2, %0) : (!emitc.opaque<"std::string">, !emitc.opaque<"std::string">) -> !emitc.opaque<"seal::Ciphertext">
    %9 = emitc.call "evaluator_multiply"(%8, %8) : (!emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
    %10 = emitc.call "evaluator_relinearize"(%9, %7) : (!emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::RelinKeys">) -> !emitc.opaque<"seal::Ciphertext">
    %11 = emitc.call "evaluator_multiply"(%10, %10) : (!emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
    %12 = emitc.call "evaluator_relinearize"(%11, %7) : (!emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::RelinKeys">) -> !emitc.opaque<"seal::Ciphertext">
    %13 = emitc.call "evaluator_multiply"(%12, %12) : (!emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
    %14 = emitc.call "evaluator_relinearize"(%13, %7) : (!emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::RelinKeys">) -> !emitc.opaque<"seal::Ciphertext">
    %15 = emitc.call "evaluator_load_ctxt"(%1, %0) : (!emitc.opaque<"std::string">, !emitc.opaque<"std::string">) -> !emitc.opaque<"seal::Ciphertext">
    %16 = emitc.call "evaluator_multiply"(%15, %15) : (!emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
    %17 = emitc.call "evaluator_relinearize"(%16, %7) : (!emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::RelinKeys">) -> !emitc.opaque<"seal::Ciphertext">
    %18 = emitc.call "evaluator_modswitch_to"(%17, %17) : (!emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
    %19 = emitc.call "evaluator_multiply"(%18, %18) : (!emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
    %20 = emitc.call "evaluator_relinearize"(%19, %7) : (!emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::RelinKeys">) -> !emitc.opaque<"seal::Ciphertext">
    %21 = emitc.call "evaluator_modswitch_to"(%20, %20) : (!emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
    %22 = emitc.call "evaluator_multiply"(%21, %21) : (!emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
    %23 = emitc.call "evaluator_relinearize"(%22, %7) : (!emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::RelinKeys">) -> !emitc.opaque<"seal::Ciphertext">
    %24 = emitc.call "evaluator_modswitch_to"(%23, %23) : (!emitc.opaque<"seal::Ciphertext">, !emitc.opaque<"seal::Ciphertext">) -> !emitc.opaque<"seal::Ciphertext">
    return %21 : !emitc.opaque<"seal::Ciphertext">
  }
}

