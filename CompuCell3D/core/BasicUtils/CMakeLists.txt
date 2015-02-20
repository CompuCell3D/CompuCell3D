# BasicUtils needs pthreads
INCLUDE(FindThreads)
INCLUDE(CheckIncludeFile)


set(dynamic_lib_extension so) #default choice
if(${CMAKE_SYSTEM_NAME} STREQUAL Linux)
   set(dynamic_lib_extension so)
   configure_file(${CMAKE_CURRENT_SOURCE_DIR}/BasicPluginManager.h.in.linux ${CMAKE_CURRENT_SOURCE_DIR}/BasicPluginManager.h)
   SET(LIBS dl)
endif(${CMAKE_SYSTEM_NAME} STREQUAL Linux)

if(${CMAKE_SYSTEM_NAME} STREQUAL Darwin)
   set(dynamic_lib_extension dylib)
   configure_file(${CMAKE_CURRENT_SOURCE_DIR}/BasicPluginManager.h.in.linux ${CMAKE_CURRENT_SOURCE_DIR}/BasicPluginManager.h)
   SET(LIBS dl)
endif(${CMAKE_SYSTEM_NAME} STREQUAL Darwin)


SET(ADDITIONAL_BASIC_UTILS_FILES )

if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
   SET(ADDITIONAL_BASIC_UTILS_FILES WindowsGlob.cpp)
   configure_file(${CMAKE_CURRENT_SOURCE_DIR}/BasicPluginManager.h.in.windows ${CMAKE_CURRENT_SOURCE_DIR}/BasicPluginManager.h)
   SET(LIBS )
endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

SET(SRCS
        BasicException.cpp
        BasicFileLocation.cpp
        BasicString.cpp
        BasicRandomNumberGenerator.cpp
        BasicDynamicClassFactory.cpp
        BasicDynamicClassNodeBase.cpp
        BasicClassGroupFactory.cpp
        BasicPluginInfo.cpp
        BasicClassRegistry.cpp
        BasicStringTable.cpp
        ${ADDITIONAL_BASIC_UTILS_FILES}
)



# If we have pthreads build BasicThread
# IF(CMAKE_USE_PTHREADS_INIT)
#   SET(SRCS ${SRCS}
#     BasicCondition.cpp
#     BasicLockable.cpp
#     BasicThread.cpp)
#   SET(LIBS ${LIBS} ${CMAKE_THREAD_LIBS_INIT})
# ENDIF(CMAKE_USE_PTHREADS_INIT)
#
# CHECK_INCLUDE_FILE(regex.h HAS_REGEX_H)
# IF(HAS_REGEX_H)
#   SET(SRCS ${SRCS} BasicRegex.cpp)
# ENDIF(HAS_REGEX_H)
#
#
#
#
# CHECK_INCLUDE_FILE(unistd.h HAS_UNISTD_H)
# IF(HAS_UNISTD_H)
#   SET(SRCS ${SRCS}
#     BasicDebugger.cpp
#     BasicPipe.cpp
#     BasicProcess.cpp)
# ENDIF(HAS_UNISTD_H)


if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
   ADD_STATIC_LIBRARY(BasicUtils ${SRCS})
else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
   ADD_SHARED_LIBRARY(BasicUtils ${SRCS} LINK_LIBRARIES ${LIBS})

endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)






INSTALL_FILES(/include/CompuCell3D/BasicUtils .h
  BasicArray
  BasicClassAccessorBase
  BasicClassAccessor
  BasicClassFactoryBase
  BasicClassFactory
  BasicClassGroupArray
  BasicClassGroupFactory
  BasicClassGroup
  BasicClassRegistry
  BasicCounter
  BasicDynamicClassFactory
  BasicDynamicClassNodeBase
  BasicDynamicClassNode
  BasicException
  BasicFileLocation
  BasicPluginInfo
  BasicPluginManager
  BasicPluginProxy
  BasicRandomNumberGenerator
  BasicSmartPointer
  BasicString
  BasicStringTable
  BasicUtils
  WindowsGlob
  wrapdll
  Zap
)
