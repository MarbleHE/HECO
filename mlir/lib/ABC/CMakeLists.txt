add_mlir_dialect_library(MLIRABC
		FHEDialect.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/ABC

        DEPENDS
        MLIRABCOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)
