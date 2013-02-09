ADD_COMPUCELL3D_PLUGIN(FocalPointPlasticity LINK_LIBRARIES ${PLUGIN_DEPENDENCIES} EXTRA_COMPILER_FLAGS ${OpenMP_CXX_FLAGS})

# ADD_COMPUCELL3D_PLUGIN(FocalPointPlasticity
  # FocalPointPlasticityPlugin.cpp
  # FocalPointPlasticityPluginProxy.cpp 
  # LINK_LIBRARIES ${PLUGIN_DEPENDENCIES}
  # EXTRA_COMPILER_FLAGS ${OpenMP_CXX_FLAGS}
  # )


# ADD_COMPUCELL3D_PLUGIN_HEADERS(FocalPointPlasticity
  # FocalPointPlasticityTracker.h
  # FocalPointPlasticityPlugin.h
  # )
