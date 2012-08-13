
ADD_COMPUCELL3D_PLUGIN(ClusterSurfaceTracker
  ClusterSurfaceTrackerPlugin.cpp
  ClusterSurfaceTrackerPluginProxy.cpp LINK_LIBRARIES ${PLUGIN_DEPENDENCIES} EXTRA_COMPILER_FLAGS ${OpenMP_CXX_FLAGS})

ADD_COMPUCELL3D_PLUGIN_HEADERS(ClusterSurfaceTracker
  ClusterSurfaceTrackerPlugin.h
  ClusterSurfaceTrackerDLLSpecifier.h
  
  )
