# Generate a list of CMake library targets so that other CMake projects can
# link against them.
set(HECO_INSTALL_PACKAGE_DIR lib${LLVM_LIBDIR_SUFFIX}/cmake/heco)
set(heco_cmake_builddir "${CMAKE_BINARY_DIR}/${HECO_INSTALL_PACKAGE_DIR}")

# Keep this in sync with mlir/cmake/CMakeLists.txt!
set(MLIR_INSTALL_PACKAGE_DIR lib${LLVM_LIBDIR_SUFFIX}/cmake/mlir)
set(mlir_cmake_builddir "${MLIR_BINARY_DIR}/${MLIR_INSTALL_PACKAGE_DIR}")

# Keep this in sync with llvm/cmake/CMakeLists.txt!
set(LLVM_INSTALL_PACKAGE_DIR lib${LLVM_LIBDIR_SUFFIX}/cmake/llvm)
set(llvm_cmake_builddir "${LLVM_BINARY_DIR}/${LLVM_INSTALL_PACKAGE_DIR}")

# Generate a list of CMake library targets so that other CMake projects can
# link against them. LLVM calls its version of this file LLVMExports.cmake, but
# the usual CMake convention seems to be ${Project}Targets.cmake.
get_property(HECO_EXPORTS GLOBAL PROPERTY HECO_EXPORTS)
export(TARGETS ${HECO_EXPORTS} FILE ${heco_cmake_builddir}/HECOTargets.cmake)

get_property(HECO_ALL_LIBS GLOBAL PROPERTY HECO_ALL_LIBS)
get_property(HECO_DIALECT_LIBS GLOBAL PROPERTY HECO_DIALECT_LIBS)
get_property(HECO_CONVERSION_LIBS GLOBAL PROPERTY HECO_CONVERSION_LIBS)
get_property(HECO_TRANSLATION_LIBS GLOBAL PROPERTY HECO_TRANSLATION_LIBS)
get_property(HECO_ANALYSIS_LIBS GLOBAL PROPERTY HECO_ANALYSIS_LIBS)

# Generate HECOConfig.cmake for the build tree.
set(HECO_CONFIG_CMAKE_DIR "${heco_cmake_builddir}")
set(HECO_CONFIG_LLVM_CMAKE_DIR "${llvm_cmake_builddir}")
set(HECO_CONFIG_MLIR_CMAKE_DIR "${mlir_cmake_builddir}")
set(HECO_CONFIG_LIBRARY_DIRS "${HECO_LIBRARY_DIR}")
set(HECO_CONFIG_BINARY_DIR "${HECO_BINARY_DIR}")
set(HECO_CONFIG_TOOLS_DIR "${HECO_TOOLS_DIR}")
set(HECO_CONFIG_INCLUDE_EXPORTS "include(\"\${HECO_CMAKE_DIR}/HECOTargets.cmake\")")
set(HECO_CONFIG_INCLUDE_DIRS
  "${HECO_SOURCE_DIR}/include"
  "${HECO_BINARY_DIR}/include"
  )
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/HECOConfig.cmake.in
  ${heco_cmake_builddir}/HECOConfig.cmake
  @ONLY)
set(HECO_CONFIG_CMAKE_DIR)
set(HECO_CONFIG_LLVM_CMAKE_DIR)
set(HECO_CONFIG_MLIR_CMAKE_DIR)
set(HECO_CONFIG_LIBRARY_DIRS)
set(HECO_CONFIG_BINARY_DIR)
set(HECO_CONFIG_TOOLS_DIR)
set(HECO_CONFIG_INCLUDE_EXPORTS)
set(HECO_CONFIG_INCLUDE_DIRS)

# Generate HECOConfig.cmake for the install tree.
set(HECO_CONFIG_CODE "
# Compute the installation prefix from this HECOConfig.cmake file location.
get_filename_component(HECO_INSTALL_PREFIX \"\${CMAKE_CURRENT_LIST_FILE}\" PATH)")
# Construct the proper number of get_filename_component(... PATH)
# calls to compute the installation prefix.
string(REGEX REPLACE "/" ";" _count "${HECO_INSTALL_PACKAGE_DIR}")
foreach(p ${_count})
  set(HECO_CONFIG_CODE "${HECO_CONFIG_CODE}
get_filename_component(HECO_INSTALL_PREFIX \"\${HECO_INSTALL_PREFIX}\" PATH)")
endforeach(p)
set(HECO_CONFIG_CMAKE_DIR "\${HECO_INSTALL_PREFIX}/${HECO_INSTALL_PACKAGE_DIR}")
set(HECO_CONFIG_LLVM_CMAKE_DIR "\${HECO_INSTALL_PREFIX}/${LLVM_INSTALL_PACKAGE_DIR}")
set(HECO_CONFIG_MLIR_CMAKE_DIR "\${HECO_INSTALL_PREFIX}/${MLIR_INSTALL_PACKAGE_DIR}")
set(HECO_CONFIG_LIBRARY_DIRS "\${HECO_INSTALL_PREFIX}/lib")
set(HECO_CONFIG_BINARY_DIR "\${HECO_INSTALL_PREFIX}")
set(HECO_CONFIG_TOOLS_DIR "\${HECO_INSTALL_PREFIX}/bin")
set(HECO_CONFIG_INCLUDE_EXPORTS "include(\"\${HECO_CMAKE_DIR}/HECOTargets.cmake\")")
set(HECO_CONFIG_INCLUDE_DIRS
  "\${HECO_INSTALL_PREFIX}/include"
  )
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/HECOConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/HECOConfig.cmake
    @ONLY)
set(HECO_CONFIG_CODE)
set(HECO_CONFIG_CMAKE_DIR)
set(HECO_CONFIG_LLVM_CMAKE_DIR)
set(HECO_CONFIG_MLIR_CMAKE_DIR)
set(HECO_CONFIG_LIBRARY_DIRS)
set(HECO_CONFIG_BINARY_DIR)
set(HECO_CONFIG_TOOLS_DIR)
set(HECO_CONFIG_INCLUDE_EXPORTS)
set(HECO_CONFIG_INCLUDE_DIRS)

if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
  # Not TOOLCHAIN ONLY, so install the HECO parts as well
  # Include the cmake files so other tools can use heco-tblgen, etc.
  install(EXPORT HECOTargets DESTINATION ${HECO_INSTALL_PACKAGE_DIR}
          COMPONENT heco-cmake-exports)

  install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/HECOConfig.cmake
    ${CMAKE_CURRENT_SOURCE_DIR}/AddHECO.cmake
    DESTINATION ${HECO_INSTALL_PACKAGE_DIR}
    COMPONENT heco-cmake-exports)

  if(NOT LLVM_ENABLE_IDE)
    # Add a dummy target so this can be used with LLVM_DISTRIBUTION_COMPONENTS
    add_custom_target(heco-cmake-exports)
    add_llvm_install_targets(install-heco-cmake-exports
                             COMPONENT heco-cmake-exports)
  endif()
endif()