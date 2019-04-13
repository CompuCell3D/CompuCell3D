#INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_PATH})
#   INCLUDE_DIRECTORIES("/usr/local/include")

# FIND_PACKAGE(PythonLibs)

INCLUDE_DIRECTORIES(${Python INCLUDE_DIRS})

ADD_COMPUCELL3D_PLUGIN(MolecularContact
  MolecularContactPlugin.cpp
  MolecularContactPluginProxy.cpp
#  MolecularContactTrackerPlugin.cpp
#  MolecularContactTrackerPluginProxy.cpp
   
  LINK_LIBRARIES ${PLUGIN_DEPENDENCIES}
# ${PYTHON_LIBRARIES}
  )

ADD_COMPUCELL3D_PLUGIN_HEADERS(MolecularContact
  MolecularContactPlugin.h
#  MolecularContactTracker.h
#  MolecularContactTrackerPlugin.h
  )
