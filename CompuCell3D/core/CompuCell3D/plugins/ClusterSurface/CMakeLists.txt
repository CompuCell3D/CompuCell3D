
ADD_COMPUCELL3D_PLUGIN(ClusterSurface
  ClusterSurfacePlugin.cpp
  ClusterSurfacePluginProxy.cpp LINK_LIBRARIES ${PLUGIN_DEPENDENCIES} EXTRA_COMPILER_FLAGS ${OpenMP_CXX_FLAGS})

ADD_COMPUCELL3D_PLUGIN_HEADERS(ClusterSurface
  ClusterSurfacePlugin.h
  ClusterSurfaceDLLSpecifier.h
  
  )
