add_heco_dialect_library(HECOBGV
        BGVDialect.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/IR/BGV

        DEPENDS
        MLIRBGVIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        )
