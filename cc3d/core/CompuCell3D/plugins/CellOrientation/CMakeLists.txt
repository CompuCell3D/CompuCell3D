ADD_COMPUCELL3D_PLUGIN(CellOrientation
  CellOrientationPlugin.cpp
  CellOrientationPluginProxy.cpp
  LINK_LIBRARIES ${PLUGIN_DEPENDENCIES}
  )

ADD_COMPUCELL3D_PLUGIN_HEADERS(CellOrientation
  CellOrientationPlugin.h
  CellOrientationVector.h
  )
