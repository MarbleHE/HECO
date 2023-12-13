set(LLVM_TARGET_DEFINITIONS WaterlineRescale.td)
mlir_tablegen(WaterlineRescale.cpp.inc -gen-rewriters)
add_public_tablegen_target(evawaterline)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_heco_conversion_library(HECOWaterlineRescale
        WaterlineRescale.cpp

        #ADDITIONAL_HEADER_DIRS
        #Passes

        DEPENDS
        evawaterline

        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        HECOEVADialect
        )

