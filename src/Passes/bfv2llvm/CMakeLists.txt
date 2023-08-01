set(LLVM_TARGET_DEFINITIONS LowerBFVToLLVM.td)
mlir_tablegen(LowerBFVToLLVM.cpp.inc -gen-rewriters)
add_public_tablegen_target(bfv2llvm)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_heco_conversion_library(HECOLowerBFVToLLVM
        LowerBFVToLLVM.cpp

        #ADDITIONAL_HEADER_DIRS
        #Passes

        DEPENDS
        bfv2llvm

        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        HECOBFVDialect
)