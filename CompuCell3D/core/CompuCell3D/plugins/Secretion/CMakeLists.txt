ADD_COMPUCELL3D_PLUGIN(Secretion LINK_LIBRARIES ${PLUGIN_DEPENDENCIES} EXTRA_COMPILER_FLAGS ${OpenMP_CXX_FLAGS}) 

# ADD_COMPUCELL3D_PLUGIN(Secretion
  # SecretionPlugin.cpp
  # FieldSecretor.cpp
  # SecretionDataP.cpp
  # SecretionPluginProxy.cpp LINK_LIBRARIES ${PLUGIN_DEPENDENCIES} EXTRA_COMPILER_FLAGS ${OpenMP_CXX_FLAGS}) 

# ADD_COMPUCELL3D_PLUGIN_HEADERS(Secretion
  # SecretionPlugin.h
  # FieldSecretor.h
  # SecretionDataP.h)
