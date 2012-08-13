
ADD_COMPUCELL3D_PLUGIN(BoundaryMonitor
  BoundaryMonitorPlugin.cpp
  BoundaryMonitorPluginProxy.cpp LINK_LIBRARIES ${PLUGIN_DEPENDENCIES} EXTRA_COMPILER_FLAGS ${OpenMP_CXX_FLAGS})

ADD_COMPUCELL3D_PLUGIN_HEADERS(BoundaryMonitor
  BoundaryMonitorPlugin.h
  BoundaryMonitorDLLSpecifier.h
  
  )
