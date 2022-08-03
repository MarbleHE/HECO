add_heco_conversion_library(HECOCatchAll
        Batching.cpp
        CombineSimplify.cpp
        InternalOperandBatching.cpp
        Nary.cpp
        ScalarBatching.cpp
        Tensor2BatchedSecret.cpp
        UnrollLoops.cpp

        #ADDITIONAL_HEADER_DIRS
        #Passes

        DEPENDS

        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        HECOFHEDialect
        )

