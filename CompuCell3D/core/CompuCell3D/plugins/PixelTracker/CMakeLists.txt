ADD_COMPUCELL3D_PLUGIN(PixelTracker
  PixelTrackerPlugin.cpp
  BoundaryPixelTrackerPlugin.cpp
  PixelTrackerPluginProxy.cpp LINK_LIBRARIES ${PLUGIN_DEPENDENCIES})


ADD_COMPUCELL3D_PLUGIN_HEADERS(PixelTracker
  PixelTracker.h
  PixelTrackerPlugin.h
  BoundaryPixelTracker.h
  BoundaryPixelTrackerPlugin.h
  )
