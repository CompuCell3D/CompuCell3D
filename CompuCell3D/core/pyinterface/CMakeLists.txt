cmake_policy(SET CMP0078 NEW)
# I don't really know why these two lines are needed!
PROJECT(PYINTERFACE)

# include(${CMAKE_CURRENT_SOURCE_DIR}/FindSWIG.cmake) # this is slightly modified FindSWIG.cmake file -  had to tune it to work with OSX

FIND_PACKAGE(SWIG REQUIRED)

# INCLUDE(${SWIG_USE_FILE})
INCLUDE(UseSWIG)

# FIND_PACKAGE(PythonLibs)
# FIND_PACKAGE(PythonInterp)


SET(NUMPY_INCLUDE_DIR )
# searching for numpy include dir
EXEC_PROGRAM(${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/core/pyinterface ARGS NumpyHeader.py
      OUTPUT_VARIABLE NUMPY_INCLUDE_DIR)

message ("NUMPY_INCLUDE_DIR = ${NUMPY_INCLUDE_DIR}")
INCLUDE_DIRECTORIES(${Python_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
INCLUDE_DIRECTORIES(${CMAKE_BINARY_DIR}/include)
INCLUDE_DIRECTORIES(${NUMPY_INCLUDE_DIR})


SET(PYINTERFACE_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/pyinterface)
SET(CMAKE_SWIG_FLAGS "")


set(PYTHON_LIBRARY_TO_LINK ${Python_LIBRARY_RELEASE})
set(undefined_link_flags " ")
if(${CMAKE_SYSTEM_NAME} STREQUAL Darwin)
    # on osx wih conda python 3 we do not link python
    # https://github.com/shogun-toolbox/shogun/issues/4068
    # Ok having spent some time on that I finally found the culprit. I had to move some lines around in generated SWIG wrapper to realise that the true error masked by this segfault is:
    #
    # PyThreadState_Get: no current thread
    # Here is the discussion I found then: https://groups.google.com/a/continuum.io/forum/#!topic/anaconda/057P4uNWyCU
    #
    # Anaconda guys do link python statically. So we run against dynamic library 'python' and the interpreter runs against statically linked 'python'. This causes stuff like segfault and 'no current thread' as that dynamic python is not initialised.
    #
    # The solution is quite simple. In case of anaconda we should not link but add -undefined dynamic_lookup instead. Patch to come soon.


    set(PYTHON_LIBRARY_TO_LINK )
    set(undefined_link_flags "-undefined dynamic_lookup")
endif()


IF(SWIG_FOUND)
    add_subdirectory(CC3DXML)
    add_subdirectory(CompuCellPython)

# if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

# else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
    # add_subdirectory(FiPyInterface)
# endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

#    add_subdirectory(PlayerPythonOld)

    FIND_PACKAGE(VTK REQUIRED)

    message("VTK_MAJOR_VERSION=" ${VTK_MAJOR_VERSION})
    if (${VTK_MAJOR_VERSION} EQUAL "6")
        add_definitions(-DVTK6)
    endif()

    if (${VTK_MAJOR_VERSION} EQUAL "7")    
        add_definitions(-DVTK6)
    endif()

    if (${VTK_MAJOR_VERSION} EQUAL "8")    
        add_definitions(-DVTK6)
    endif()


    add_subdirectory(PlayerPythonNew)
    add_subdirectory(SerializerDE)
    add_subdirectory(PyPlugin)
    # add_subdirectory(Example)
    # add_subdirectory(Fields)
if(USE_DOLFIN)
    add_subdirectory(dolfinCC3D)
endif(USE_DOLFIN)

ELSE(SWIG_FOUND)
    message("could not find SWIG in your system")
ENDIF(SWIG_FOUND)

#CMAKE_MINIMUM_REQUIRED(VERSION 2.2)
#PROJECT(PYINTERFACE)
