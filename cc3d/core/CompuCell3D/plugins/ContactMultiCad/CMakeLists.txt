ADD_COMPUCELL3D_PLUGIN(ContactMultiCad
  ContactMultiCadPlugin.cpp
  ContactMultiCadPluginProxy.cpp LINK_LIBRARIES ${PLUGIN_DEPENDENCIES})

ADD_COMPUCELL3D_PLUGIN_HEADERS(ContactMultiCad
  ContactMultiCadPlugin.h)
