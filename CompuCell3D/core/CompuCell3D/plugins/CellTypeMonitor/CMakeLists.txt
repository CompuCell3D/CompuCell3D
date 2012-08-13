
ADD_COMPUCELL3D_PLUGIN(CellTypeMonitor
  CellTypeMonitorPlugin.cpp
  CellTypeMonitorPluginProxy.cpp LINK_LIBRARIES ${PLUGIN_DEPENDENCIES} EXTRA_COMPILER_FLAGS ${OpenMP_CXX_FLAGS})

ADD_COMPUCELL3D_PLUGIN_HEADERS(CellTypeMonitor
  CellTypeMonitorPlugin.h
  CellTypeMonitorDLLSpecifier.h
  
  )
