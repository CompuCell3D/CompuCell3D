ADD_COMPUCELL3D_PLUGIN(ConvergentExtension
  ConvergentExtensionPlugin.cpp
  ConvergentExtensionPluginProxy.cpp LINK_LIBRARIES ${PLUGIN_DEPENDENCIES})


ADD_COMPUCELL3D_PLUGIN_HEADERS(ConvergentExtension
  ConvergentExtensionPlugin.h)
