ADD_COMPUCELL3D_PLUGIN(ContactLocalProduct
  ContactLocalProductPlugin.cpp
  ContactLocalProductPluginProxy.cpp LINK_LIBRARIES ${PLUGIN_DEPENDENCIES} EXTRA_COMPILER_FLAGS ${OpenMP_CXX_FLAGS})


ADD_COMPUCELL3D_PLUGIN_HEADERS(ContactLocalProduct
  ContactLocalProductPlugin.h)
