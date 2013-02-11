ADD_COMPUCELL3D_PLUGIN(Plasticity LINK_LIBRARIES ${PLUGIN_DEPENDENCIES} EXTRA_COMPILER_FLAGS ${OpenMP_CXX_FLAGS})

# ADD_COMPUCELL3D_PLUGIN(Plasticity
  # PlasticityPlugin.cpp
  # PlasticityPluginProxy.cpp
  # PlasticityTrackerPlugin.cpp
  # PlasticityTrackerPluginProxy.cpp
  # LINK_LIBRARIES ${PLUGIN_DEPENDENCIES}
  # EXTRA_COMPILER_FLAGS ${OpenMP_CXX_FLAGS}
  # )

# ADD_COMPUCELL3D_PLUGIN_HEADERS(Plasticity
  # PlasticityPlugin.h
  # PlasticityTracker.h
  # PlasticityTrackerPlugin.h
  # )
