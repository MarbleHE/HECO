set(LLVM_TARGET_DEFINITIONS LowerFHEToBGV.td)
mlir_tablegen(LowerFHEToBGV.cpp.inc -gen-rewriters)
add_public_tablegen_target(fhe2bgv)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_heco_conversion_library(HECOLowerFHEToBGV
        LowerFHEToBGV.cpp

        #ADDITIONAL_HEADER_DIRS
        #Passes

        DEPENDS
        fhe2bgv

        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        HECOFHEDialect
        HECOBGVDialect
        )

