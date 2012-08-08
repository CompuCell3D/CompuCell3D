
ADD_COMPUCELL3D_PLUGIN(PlayerSettings PlayerSettingsPlugin.cpp PlayerSettingsPluginProxy.cpp LINK_LIBRARIES ${PLUGIN_DEPENDENCIES})


ADD_COMPUCELL3D_PLUGIN_HEADERS(PlayerSettings
  PlayerSettings.h
  PlayerSettingsPlugin.h
  )
