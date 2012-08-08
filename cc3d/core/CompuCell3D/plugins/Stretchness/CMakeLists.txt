ADD_COMPUCELL3D_PLUGIN(Stretchness
  StretchnessPlugin.cpp
  StretchnessPluginProxy.cpp LINK_LIBRARIES ${PLUGIN_DEPENDENCIES})

ADD_COMPUCELL3D_PLUGIN_HEADERS(Stretchness
  StretchnessPlugin.h)
