set(LLVM_TARGET_DEFINITIONS LowerFHEToEVA.td)
mlir_tablegen(LowerFHEToEVA.cpp.inc -gen-rewriters)
add_public_tablegen_target(fhe2eva)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_heco_conversion_library(HECOLowerFHEToEVA
        LowerFHEToEVA.cpp

        #ADDITIONAL_HEADER_DIRS
        #Passes

        DEPENDS
        fhe2eva

        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        HECOFHEDialect
        HECOEVADialect
        )

