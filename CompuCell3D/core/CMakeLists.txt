# todo - remove it from here
# if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
#     # FIND_PACKAGE(PythonInterpreter)
#     FIND_PACKAGE(PythonLibs)
#     FIND_PACKAGE(PythonInterp)
#     # INCLUDE(FindPythonLibs)
#     set(Python_EXECUTABLE ${PYTHON_EXECUTABLE} CACHE PATH "Python executable" FORCE)
#     set(Python_LIBRARIES ${PYTHON_LIBRARY} CACHE PATH "Python library" FORCE)
#     set(Python_INCLUDE_DIRS ${PYTHON_INCLUDE_DIR} CACHE PATH "Python include directory" FORCE )
#     set(Python_LIBRARY_RELEASE ${PYTHON_LIBRARY} CACHE PATH "Python release library directory" FORCE )
#
# else()
#     # find_package (Python 3 COMPONENTS Interpreter Development)
#     FIND_PACKAGE(PythonLibs)
#     FIND_PACKAGE(PythonInterp)
#     # INCLUDE(FindPythonLibs)
#     set(Python_EXECUTABLE ${PYTHON_EXECUTABLE} CACHE PATH "Python executable" FORCE)
#     set(Python_LIBRARIES ${PYTHON_LIBRARY} CACHE PATH "Python library" FORCE)
#     set(Python_INCLUDE_DIRS ${PYTHON_INCLUDE_DIR} CACHE PATH "Python include directory" FORCE )
#     set(Python_LIBRARY_RELEASE ${PYTHON_LIBRARY} CACHE PATH "Python release library directory" FORCE )
#
#
# endif()
#
# # find_package (Python 3 COMPONENTS Interpreter Development)
#
#
# message("Python_FOUND: ${Python_FOUND}")
# message("Python_Interpreter_FOUND: ${Python_Interpreter_FOUND}")
# message("Python_VERSION: ${Python_VERSION}")
# message("Python_Development_FOUND: ${Python_Development_FOUND}")
# message("Python_EXECUTABLE: ${Python_EXECUTABLE}")
#
# message("Python_Development_FOUND: ${Python_Development_FOUND}")
# message("Python_INCLUDE_DIRS: ${Python_INCLUDE_DIRS}")
# message("Python_LIBRARIES: ${Python_LIBRARIES}")
# message("Python_LIBRARY_DIRS: ${Python_LIBRARY_DIRS}")
# message("Python_RUNTIME_LIBRARY_DIRS: ${Python_RUNTIME_LIBRARY_DIRS}")

INCLUDE_DIRECTORIES (
    ${COMPUCELL3D_SOURCE_DIR}/core
    )


ADD_SUBDIRECTORY(BasicUtils)
ADD_SUBDIRECTORY(Utils)
ADD_SUBDIRECTORY(XMLUtils)
ADD_SUBDIRECTORY(PublicUtilities)
ADD_SUBDIRECTORY(muParser)
ADD_SUBDIRECTORY(MaBoSS)
ADD_SUBDIRECTORY(Cleaver)
ADD_SUBDIRECTORY(CompuCell3D)
# # ADD_SUBDIRECTORY(CompuCellPlayer)

IF(BUILD_PYINTERFACE)
    ADD_SUBDIRECTORY(pyinterface)
ENDIF(BUILD_PYINTERFACE)

ADD_SUBDIRECTORY(post_install_tasks)








