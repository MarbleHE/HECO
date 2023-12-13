get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
message(  ${dialect_libs} )
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIREmitCDialect
        MLIROptLib
        HECOFHEDialect
        HECOEVADialect
        HECOBFVDialect
        HECOPolyDialect
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
        MLIRInferTypeOpInterface
        MLIRControlFlowInterfaces
        MLIRSideEffectInterfaces
        MLIRTargetLLVMIRExport
        MLIRTransforms  
        HECOLowerBFVToEmitC
        HECOLowerBFVToLLVM
        HECOLowerFHEToBFV
        HECOLowerFHEToEmitC
        HECOCatchAll
        )
add_llvm_executable(heco heco.cpp)

llvm_update_compile_flags(heco)
target_link_libraries(heco PRIVATE ${LIBS})

add_dependencies(heco bfv2emitc)

mlir_check_all_link_libraries(heco)
