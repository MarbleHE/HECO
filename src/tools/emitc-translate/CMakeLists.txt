get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
set(LIBS
        ${dialect_libs}
        ${translation_libs}
        MLIROptLib
        MLIRAnalysis
        MLIRCallInterfaces
        MLIRCastInterfaces
        MLIRExecutionEngine
        MLIRIR
        MLIRLLVMCommonConversion
        MLIRLLVMToLLVMIRTranslation
        MLIRMemRefDialect
        MLIRParser
        MLIRPass
        MLIRTargetCpp
        MLIRTranslateLib
        MLIRSideEffectInterfaces
        MLIRTargetLLVMIRExport
        MLIRTransforms
        )
add_llvm_executable(emitc-translate
        emitc-translate.cpp)

llvm_update_compile_flags(emitc-translate)
target_link_libraries(emitc-translate PUBLIC ${LIBS})

mlir_check_all_link_libraries(emitc-translate)
