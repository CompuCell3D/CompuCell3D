ADD_COMPUCELL3D_PLUGIN(FocalPointContact
  FocalPointBoundaryPixelTrackerPlugin.cpp
  FocalPointContactPlugin.cpp
  FocalPointBoundaryPixelTracker.cpp
  FocalPointContactPluginProxy.cpp 
  LINK_LIBRARIES ${PLUGIN_DEPENDENCIES}
  EXTRA_COMPILER_FLAGS ${OpenMP_CXX_FLAGS}
  )


ADD_COMPUCELL3D_PLUGIN_HEADERS(FocalPointContact
  FocalPointBoundaryPixelTracker.h
  FocalPointContactPlugin.h
  FocalPointBoundaryPixelTrackerPlugin.h
  )
