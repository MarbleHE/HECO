set(LLVM_TARGET_DEFINITIONS LowerFHEToBFV.td)
mlir_tablegen(LowerFHEToBFV.cpp.inc -gen-rewriters)
add_public_tablegen_target(fhe2bfv)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_heco_conversion_library(HECOLowerFHEToBFV
        LowerFHEToBFV.cpp

        #ADDITIONAL_HEADER_DIRS
        #Passes

        DEPENDS
        fhe2bfv

        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        HECOFHEDialect
        HECOBFVDialect
        )

