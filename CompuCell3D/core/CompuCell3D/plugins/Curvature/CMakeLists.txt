ADD_COMPUCELL3D_PLUGIN(Curvature
  CurvaturePlugin.cpp
  CurvaturePluginProxy.cpp 
  LINK_LIBRARIES ${PLUGIN_DEPENDENCIES}
  EXTRA_COMPILER_FLAGS ${OpenMP_CXX_FLAGS}
  )


ADD_COMPUCELL3D_PLUGIN_HEADERS(Curvature
  CurvatureTracker.h
  CurvaturePlugin.h
  )
