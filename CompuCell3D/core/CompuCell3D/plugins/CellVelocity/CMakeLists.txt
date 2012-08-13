ADD_COMPUCELL3D_PLUGIN(CellVelocity
  CellInstantVelocityPlugin.cpp
  CellVelocityData.cpp
  CellVelocityPluginProxy.cpp
  CellInstantVelocityPluginProxy.cpp
  CellVelocityPlugin.cpp)

ADD_COMPUCELL3D_PLUGIN_HEADERS(CellVelocity
  CellInstantVelocityPlugin.h
  CellVelocityDataAccessor.h
  CellVelocityDataClassFactory.h
  CellVelocityData.h
  CellVelocityPlugin.h
  InstantVelocityData.h)
