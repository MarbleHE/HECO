add_mlir_dialect_library(MLIRABC
        ABCDialect.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/ABC

        DEPENDS
        MLIRABCIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)
