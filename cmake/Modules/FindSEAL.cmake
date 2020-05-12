find_path(SEAL_INCLUDE_DIR NAMES seal/seal.h PATH_SUFFIXES SEAL-3.4)
find_library(SEAL_LIBRARY NAMES seal seal-3.4)


set(SEAL_INCLUDE_DIRS ${SEAL_INCLUDE_DIR})
set(SEAL_LIBRARIES ${SEAL_LIBRARY})

find_package(ZLIB 1.2.11 EXACT)
if(ZLIB_FOUND)
    set(SEAL_LIBRARIES ${SEAL_LIBRARY} ${ZLIB_LIBRARIES})
    include_directories(SYSTEM ${ZLIB_INCLUDE_DIRS})
endif()

find_package(MSGSL MODULE)
if(MSGSL_FOUND)
    include_directories(SYSTEM ${MSGSL_INCLUDE_DIRS})
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SEAL DEFAULT_MSG
        SEAL_LIBRARIES SEAL_INCLUDE_DIRS)

mark_as_advanced(SEAL_LIBRARY SEAL_INCLUDE_DIR)

