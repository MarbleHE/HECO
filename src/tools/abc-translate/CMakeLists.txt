get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
set(LIBS
        ${dialect_libs}
        ${translation_libs}
        MLIROptLib
        MLIRABC
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
        MLIRTargetCpp
        MLIRTranslation
        MLIRSideEffectInterfaces
        MLIRTargetLLVMIRExport
        MLIRTransforms
        )
add_llvm_executable(abc-translate
        ${PROJECT_SOURCE_DIR}/src/tools/abc-translate/abc-translate.cpp)

llvm_update_compile_flags(abc-translate)
target_link_libraries(abc-translate PRIVATE ${LIBS})

mlir_check_all_link_libraries(abc-translate)
