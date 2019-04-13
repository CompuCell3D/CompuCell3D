

find_package (Python 3 COMPONENTS Interpreter Development)



message("Python_FOUND: ${Python_FOUND}")
message("Python_Interpreter_FOUND: ${Python_Interpreter_FOUND}")
message("Python_VERSION: ${Python_VERSION}")
message("Python_Development_FOUND: ${Python_Development_FOUND}")
message("Python_EXECUTABLE: ${Python_EXECUTABLE}")

message("Python_Development_FOUND: ${Python_Development_FOUND}")
message("Python_INCLUDE_DIRS: ${Python_INCLUDE_DIRS}")
message("Python_LIBRARIES: ${Python_LIBRARIES}")
message("Python_LIBRARY_DIRS: ${Python_LIBRARY_DIRS}")
message("Python_RUNTIME_LIBRARY_DIRS: ${Python_RUNTIME_LIBRARY_DIRS}")

INCLUDE_DIRECTORIES (
    ${COMPUCELL3D_SOURCE_DIR}/core
    )


ADD_SUBDIRECTORY(BasicUtils)
ADD_SUBDIRECTORY(Utils)
ADD_SUBDIRECTORY(XMLUtils)
ADD_SUBDIRECTORY(PublicUtilities)
ADD_SUBDIRECTORY(muParser)
ADD_SUBDIRECTORY(Cleaver)
ADD_SUBDIRECTORY(CompuCell3D)
# # ADD_SUBDIRECTORY(CompuCellPlayer)

IF(BUILD_PYINTERFACE)
    ADD_SUBDIRECTORY(pyinterface)
ENDIF(BUILD_PYINTERFACE)

ADD_SUBDIRECTORY(post_install_tasks)








