add_heco_dialect_library(HECOBGVDialect
        BGVDialect.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/IR/BGV

        DEPENDS
        MLIRBGVIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        )
