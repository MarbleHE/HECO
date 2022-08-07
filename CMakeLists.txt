cmake_minimum_required(VERSION 3.19)

# Required for selecting MSVC runtime library
if (MSVC)
    cmake_policy(SET CMP0091 NEW)
endif ()

# MLIR/LLVM currently (release 15-rc1) uses DEPFILE in their cmake, which has changed in new versions
cmake_policy(SET CMP0116 OLD)

#########################
# Project Configuration #
#########################

# Only define a project when this CMakeLists.txt is in the "root" (i.e., this is a standtalone build)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
        project(HECO
                VERSION 0.1
                DESCRIPTION "MLIR-based compiler for Fully Homomorphic Encryption (FHE) compilation."
                LANGUAGES CXX C)
        set(HECO_STANDALONE_BUILD TRUE)
endif()

######################
# SEAL Configuration #
######################

find_package(SEAL 4.0 CONFIG) # sets SEAL_FOUND variable

###########################
# MLIR/LLVM Configuration #
###########################
if(HECO_STANDALONE_BUILD)

        set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

        # Locate MLIR, which recursively locates LLVM
        find_package(MLIR REQUIRED CONFIG)
        message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
        message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

        set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
        set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)

        list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
        list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

        include(TableGen)
        include(AddLLVM)
        include(AddMLIR)
        include(HandleLLVMOptions)

        include_directories(${LLVM_INCLUDE_DIRS})
        include_directories(${MLIR_INCLUDE_DIRS})
        include_directories(${PROJECT_SOURCE_DIR}/include)
        include_directories(${PROJECT_BINARY_DIR}/include)
        link_directories(${LLVM_BUILD_LIBRARY_DIR})
        add_definitions(${LLVM_DEFINITIONS})

        #TODO (Implementation) Handling of unit-tests, e.g. like in CIRCT https://github.com/llvm/circt/blob/fe1ddfc6e3cd2af7d4fa333897d2a4da8d4521f4/CMakeLists.txt#L84-L121

        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY
                "${CMAKE_CURRENT_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX}")
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin")

else() # Not a standalone build
        #TODO Test out nested builds!
        message(WARNING "Non-Standalone build for HECO is untested/unsupported.")
endif()


######################
# HECO Configuration #
######################

list(APPEND CMAKE_MODULE_PATH "${MLIR_MAIN_SRC_DIR}/cmake/modules")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")
include(AddHECO)

# Installing the headers and docs needs to depend on generating any public
# tablegen'd targets.
add_custom_target(heco-headers)
set_target_properties(heco-headers PROPERTIES FOLDER "Misc")
add_custom_target(heco-doc)

# Add MLIR and LLVM headers to the include path
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

# Add HECO files to the include path
include_directories(${HECO_MAIN_INCLUDE_DIR})
include_directories(${HECO_INCLUDE_DIR})


##############################
# Compiler Setup (esp. MSVC)
##############################

# Global flags aren't good CMAKE style, but these should work across pretty much all compilers
set(CXXFLAGS "--coverage --pedantic-errors -Wall -WError")

set(CMAKE_CXX_STANDARD 14 CACHE STRING "C++ standard to conform to")

if (MSVC)
    message("Using MSVC")

    # Force dynamic MSVC runtime library on Windows
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

    # Switch MSVC into multi threaded mode
    add_compile_options(/MP)

    # Force MSVC to use slightly less broken template rules
    add_compile_options(/permissive-)

    # Force MSVC to conform to C++ standard's idea of exceptions
    add_compile_options(/EHsc)

    # Try using variants
    add_compile_options(/std:c++17)

    # Don't force fopen_s (which is a MSVC-only thing)
    add_compile_definitions(_CRT_SECURE_NO_DEPRECATE)

    # Mark system headers as external
    # Treats all #include <..> as system includes, so requires some care from developers
    add_compile_options(/experimental:external /external:anglebrackets /external:W0)

endif ()


###################
# Directory Setup #
###################
add_subdirectory(include/heco)
add_subdirectory(src)


# option(HECO_INCLUDE_DOCS "Generate build targets for the HECO docs.")
# if (HECO_INCLUDE_DOCS)
#   add_subdirectory(docs)
# endif()

install(DIRECTORY include/heco
  DESTINATION include
  COMPONENT heco-headers
  FILES_MATCHING
  PATTERN "*.def"
  PATTERN "*.h"
  PATTERN "*.inc"
  PATTERN "*.td"
  PATTERN "*.sv"
  PATTERN "LICENSE.TXT"
  )

install(DIRECTORY ${HECO_INCLUDE_DIR}/heco
  DESTINATION include
  COMPONENT heco-headers
  FILES_MATCHING
  PATTERN "*.def"
  PATTERN "*.h"
  PATTERN "*.gen"
  PATTERN "*.inc"
  PATTERN "*.td"
  PATTERN "CMakeFiles" EXCLUDE
  PATTERN "config.h" EXCLUDE
  )

if (NOT LLVM_ENABLE_IDE)
  add_llvm_install_targets(install-heco-headers
                           DEPENDS heco-headers
                           COMPONENT heco-headers)
endif()

add_subdirectory(cmake/modules)

# Set RPATH to $ORIGIN on all targets.
function(set_rpath_all_targets dir)
  get_property(subdirectories DIRECTORY ${dir} PROPERTY SUBDIRECTORIES)
  foreach(subdir ${subdirectories})
    set_rpath_all_targets(${subdir})
  endforeach()

  get_directory_property(LCL_TARGETS DIRECTORY ${dir} BUILDSYSTEM_TARGETS)
  set_property(TARGET ${LCL_TARGETS} PROPERTY INSTALL_RPATH "$ORIGIN/../lib")
endfunction()

option(STANDALONE_INSTALL "Create an 'install' for packaging which doesn't \
         require installation" off)
if (STANDALONE_INSTALL)
  message(STATUS "Setting an $ORIGIN-based RPATH on all executables")
  set_rpath_all_targets(${CMAKE_CURRENT_SOURCE_DIR})
endif()


####################################################################################
# Python Frontend
#
# Use pybind11 as C++ wrapper for the Python frontend
####################################################################################

# Install pyabc package (for dev)
find_package (Python3 3.10 COMPONENTS Interpreter Development)

if(NOT Python3_Interpreter_FOUND)
    message(WARNING "The HECO python frontend requires Python 3, but no installation
        was found. Skipping frontend building...")
endif()

# Set to test the python frontend. On by default.
if(NOT PYTHON_FRONTEND_DEV)
    set(PYTHON_FRONTEND_DEV true)
endif()

# Set to build the python frontend (installing pybind11 and building the wrapper). On by default.
if(NOT PYTHON_FRONTEND_BUILD)
    set(PYTHON_FRONTEND_BUILD true)
endif()

if(PYTHON_FRONTEND_BUILD)
    include(FetchContent)
    set(JSON_BuildTests OFF CACHE INTERNAL "")
    set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
    FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG v2.10.0)

    FetchContent_GetProperties(pybind11)
    if(NOT pybind11_POPULATED)
        FetchContent_Populate(pybind11)
        add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
    endif()
    add_subdirectory(python)
endif()


################
# Installation #
################

include(GNUInstallDirs)


####################################
## TESTING
##
# added only if this is the root project
####################################
if (HECO_STANDALONE_BUILD)
    message("Enabling HECO Tests")
    enable_testing()
    add_subdirectory(test)
endif ()

##############################
# TARGET: benchmarks
#
# A collection of benchmarks
# added only if this is the root project
##############################
if (HECO_STANDALONE_BUILD)
  if (SEAL_FOUND)
      add_subdirectory(test/bench)
      add_subdirectory(test/IR/BGV)
  endif (SEAL_FOUND)
endif ()