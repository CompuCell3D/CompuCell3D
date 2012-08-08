INCLUDE_DIRECTORIES (
    ${COMPUCELL3D_SOURCE_DIR}/core
    ${COMPUCELL3D_SOURCE_DIR}/core/CompuCell3D/plugins
    )

ADD_COMPUCELL3D_PLUGIN(Rearrangement
  RearrangementPlugin.cpp
  RearrangementPluginProxy.cpp LINK_LIBRARIES ${PLUGIN_DEPENDENCIES})

ADD_COMPUCELL3D_PLUGIN_HEADERS(Rearrangement
  RearrangementPlugin.h)
