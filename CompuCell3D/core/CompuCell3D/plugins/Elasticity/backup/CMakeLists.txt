ADD_COMPUCELL3D_PLUGIN(Elasticity LINK_LIBRARIES ${PLUGIN_DEPENDENCIES} EXTRA_COMPILER_FLAGS ${OpenMP_CXX_FLAGS})

# ADD_COMPUCELL3D_PLUGIN(Elasticity
  # ElasticityPlugin.cpp
  # ElasticityPluginProxy.cpp
  # ElasticityTrackerPlugin.cpp
  # ElasticityTrackerPluginProxy.cpp
  # LINK_LIBRARIES ${PLUGIN_DEPENDENCIES}
  # EXTRA_COMPILER_FLAGS ${OpenMP_CXX_FLAGS}
  # )

# ADD_COMPUCELL3D_PLUGIN_HEADERS(Elasticity
  # ElasticityPlugin.h
  # ElasticityTracker.h
  # ElasticityTrackerPlugin.h
  # )
