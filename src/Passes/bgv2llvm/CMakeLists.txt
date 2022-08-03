set(LLVM_TARGET_DEFINITIONS LowerBGVToLLVM.td)
mlir_tablegen(LowerBGVToLLVM.cpp.inc -gen-rewriters)
add_public_tablegen_target(bgv2llvm)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_heco_conversion_library(HECOLowerBGVToLLVM
        LowerBGVToLLVM.cpp

        #ADDITIONAL_HEADER_DIRS
        #Passes

        DEPENDS
        bgv2llvm

        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        HECOBGVDialect
)