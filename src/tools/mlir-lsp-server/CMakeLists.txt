get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LLVM_LINK_COMPONENTS
        Core
        Support
        AsmParser
        )

set(LIBS
        MLIRLspServerLib
        ${dialect_libs}
        ${conversion_libs}
        HECOFHEDialect
        HECOBFVDialect
        HECOPolyDialect
        MLIRAnalysis
        MLIRDialect
        MLIRLspServerLib
        MLIRParser
        MLIRPass
        MLIRTransforms
        MLIRTransformUtils
        MLIRSupport
        MLIRIR
        )


add_llvm_tool(mlir-lsp-server-custom
        mlir-lsp-server.cpp

        DEPENDS
        ${LIBS}
        )
target_link_libraries(mlir-lsp-server-custom PRIVATE ${LIBS})
llvm_update_compile_flags(mlir-lsp-server-custom)

mlir_check_all_link_libraries(mlir-lsp-server-custom)
