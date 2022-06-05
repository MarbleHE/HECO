set(LLVM_TARGET_DEFINITIONS LowerASTtoSSA.td)
mlir_tablegen(LowerASTtoSSA.cpp.inc -gen-rewriters)
add_public_tablegen_target(ast2ssa)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_heco_conversion_library(HECOLowerAST2SSA
        LowerASTtoSSA.cpp

        #ADDITIONAL_HEADER_DIRS
        #Passes

        DEPENDS
        ast2ssa

        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        HECOAST
        HECOFHE
        )

