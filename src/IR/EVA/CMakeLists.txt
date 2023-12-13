add_heco_dialect_library(HECOEVADialect
        EVADialect.cpp
        EVAAnalyses.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/EVA

        DEPENDS
        MLIREVAIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        )
