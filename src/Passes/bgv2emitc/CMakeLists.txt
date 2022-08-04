set(LLVM_TARGET_DEFINITIONS LowerBGVToEmitC.td)
mlir_tablegen(LowerBGVToEmitC.cpp.inc -gen-rewriters)
add_public_tablegen_target(bgv2emitc)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_heco_conversion_library(HECOLowerBGVToEmitC
        LowerBGVToEmitC.cpp

        #ADDITIONAL_HEADER_DIRS
        #Passes

        DEPENDS
        bgv2emitc

        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        HECOBGVDialect
        MLIREmitCDialect
        )

