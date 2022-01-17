get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
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
        MLIRSideEffectInterfaces
        MLIRTargetLLVMIRExport
        MLIRTransforms
        )
add_llvm_executable(abc-opt
        ${PROJECT_SOURCE_DIR}/src/Passes/ast2ssa/LowerASTtoSSA.cpp
        ${PROJECT_SOURCE_DIR}/src/Passes/ssa2ssa/UnrollLoops.cpp
        ${PROJECT_SOURCE_DIR}/src/Passes/ssa2ssa/Nary.cpp
        ${PROJECT_SOURCE_DIR}/src/Passes/ssa2ssa/Tensor2BatchedSecret.cpp
        ${PROJECT_SOURCE_DIR}/src/Passes/ssa2ssa/Batching.cpp
        ${PROJECT_SOURCE_DIR}/src/Passes/ssa2ssa/InternalOperandBatching.cpp
        ${PROJECT_SOURCE_DIR}/src/Passes/ssa2cpp/LowerToEmitC.cpp
        ${PROJECT_SOURCE_DIR}/src/tools/abc-opt/abc-opt.cpp)

llvm_update_compile_flags(abc-opt)
target_link_libraries(abc-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(abc-opt)
