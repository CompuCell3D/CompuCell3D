ADD_COMPUCELL3D_PLUGIN(ContactInternal
  ContactInternalPlugin.cpp
  ContactInternalPluginProxy.cpp LINK_LIBRARIES ${PLUGIN_DEPENDENCIES})


ADD_COMPUCELL3D_PLUGIN_HEADERS(ContactInternal
  ContactInternalPlugin.h)
