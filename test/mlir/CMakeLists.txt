configure_lit_site_cfg(
        lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        lit.cfg.py
)

set(STANDALONE_TEST_DEPENDS
        FileCheck count not
        abc-opt
        abc-translate
        )

add_lit_testsuite(check-abc "Running the abc regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${STANDALONE_TEST_DEPENDS}
        )
set_target_properties(check-abc PROPERTIES FOLDER "Tests")

add_lit_testsuites(../ABC . DEPENDS ${STANDALONE_TEST_DEPENDS})


##############################
# TARGET: testing-python
##############################
if(PYTHON_FRONTEND_DEV)
    # Install pytest
    execute_process(COMMAND ${Python3_EXECUTABLE} -m pip install --user pytest)

    set(PYTEST_CMD Python3::Interpreter -m pytest --ignore=extern --ignore ${CMAKE_BINARY_DIR})

    add_test(NAME PythonFrontendSimple
            COMMAND ${PYTEST_CMD} ${PROJECT_SOURCE_DIR}/test/python_frontend/test_simple.py
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/mlir/python)

    add_test(NAME PythonFrontendHammingDistance
            COMMAND ${PYTEST_CMD} ${PROJECT_SOURCE_DIR}/test/python_frontend/test_hamming_distance.py
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/mlir/python)

    add_test(NAME PythonMatrixVectorProduct
            COMMAND ${PYTEST_CMD} ${PROJECT_SOURCE_DIR}/test/python_frontend/test_matrix_vector_product.py
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/mlir/python)

    add_test(NAME PythonSecretTainting
            COMMAND ${PYTEST_CMD} ${PROJECT_SOURCE_DIR}/test/python_frontend/test_secret_tainting.py
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/mlir/python)

    add_test(NAME PythonExternalFn
            COMMAND ${PYTEST_CMD} ${PROJECT_SOURCE_DIR}/test/python_frontend/test_external_fn.py
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/mlir/python)

    add_test(NAME PythonMLIRTranslation
            COMMAND ${PYTEST_CMD} ${PROJECT_SOURCE_DIR}/test/python_frontend/test_mlir_translation.py
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/mlir/python)

endif()