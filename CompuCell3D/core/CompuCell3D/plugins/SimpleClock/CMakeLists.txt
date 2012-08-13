ADD_COMPUCELL3D_PLUGIN(SimpleClock
  SimpleClockPlugin.cpp
  SimpleClockPluginProxy.cpp LINK_LIBRARIES ${PLUGIN_DEPENDENCIES})

ADD_COMPUCELL3D_PLUGIN_HEADERS(SimpleClock
  SimpleClock.h
  SimpleClockPlugin.h)
