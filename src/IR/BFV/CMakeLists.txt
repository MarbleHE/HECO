add_heco_dialect_library(HECOBFVDialect
        BFVDialect.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/IR/BFV

        DEPENDS
        MLIRBFVIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        )
