set(LLVM_TARGET_DEFINITIONS LowerBFVToEmitC.td)
mlir_tablegen(LowerBFVToEmitC.cpp.inc -gen-rewriters)
add_public_tablegen_target(bfv2emitc)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_heco_conversion_library(HECOLowerBFVToEmitC
        LowerBFVToEmitC.cpp

        #ADDITIONAL_HEADER_DIRS
        #Passes

        DEPENDS
        bfv2emitc

        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        HECOBFVDialect
        MLIREmitCDialect
        )

