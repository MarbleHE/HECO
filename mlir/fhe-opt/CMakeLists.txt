get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIRFHE
        MLIRAnalysis
        MLIRCallInterfaces
        MLIRCastInterfaces
        MLIRExecutionEngine
        MLIRIR
        MLIRLLVMCommonConversion
        MLIRLLVMToLLVMIRTranslation
        MLIRMemRef
        MLIRParser
        MLIRPass
        MLIRSideEffectInterfaces
        MLIRTargetLLVMIRExport
        MLIRTransforms
        )
add_llvm_executable(fhe-opt fhe-opt.cpp)

llvm_update_compile_flags(fhe-opt)
target_link_libraries(fhe-opt PRIVATE ${LIBS})

set(LLVM_TARGET_DEFINITIONS LowerFHEtoPoly.td)
mlir_tablegen(LowerFHEtoPoly.inc -gen-rewriters)
add_public_tablegen_target(fhe2poly)
add_dependencies(fhe-opt fhe2poly)


mlir_check_all_link_libraries(fhe-opt)
