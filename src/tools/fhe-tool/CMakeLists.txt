get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
message(  ${dialect_libs} )
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIREmitC
        MLIROptLib
        HECOAST
        HECOFHE
        HECOBGV
        HECOPoly
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
        MLIRInferTypeOpInterface
        MLIRControlFlowInterfaces
        MLIRSideEffectInterfaces
        MLIRTargetLLVMIRExport
        MLIRTransforms  
        HECOLowerast2hir
        HECOLowerBGVToEmitC
        HECOLowerBGVToLLVM
        HECOLowerFHEToBGV
        HECOCatchAll
        )
add_llvm_executable(fhe-tool fhe-tool.cpp)

llvm_update_compile_flags(fhe-tool)
target_link_libraries(fhe-tool PRIVATE ${LIBS})

add_dependencies(fhe-tool ast2hir)
add_dependencies(fhe-tool bgv2emitc)

mlir_check_all_link_libraries(fhe-tool)
