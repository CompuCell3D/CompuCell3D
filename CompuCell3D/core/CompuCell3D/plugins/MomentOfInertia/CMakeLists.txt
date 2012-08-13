SET(SRCS
  MomentOfInertiaPlugin.cpp
  MomentOfInertiaPluginProxy.cpp
  )

ADD_COMPUCELL3D_PLUGIN(MomentOfInertia
  MomentOfInertiaPlugin.cpp
  MomentOfInertiaPluginProxy.cpp
  LINK_LIBRARIES ${PLUGIN_DEPENDENCIES}
  )

ADD_COMPUCELL3D_PLUGIN_HEADERS(MomentOfInertia
  CellMomentOfInertia.h
  MomentOfInertiaPlugin.h
  )
