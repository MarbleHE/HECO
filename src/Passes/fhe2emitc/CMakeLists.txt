set(LLVM_TARGET_DEFINITIONS LowerFHEToEmitC.td)
mlir_tablegen(LowerFHEToEmitC.cpp.inc -gen-rewriters)
add_public_tablegen_target(fhe2emitc)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_heco_conversion_library(HECOLowerFHEToEmitC
        LowerFHEToEmitC.cpp

        #ADDITIONAL_HEADER_DIRS
        #Passes

        DEPENDS
        fhe2emitc

        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        HECOFHEDialect
        MLIREmitCDialect
        )

