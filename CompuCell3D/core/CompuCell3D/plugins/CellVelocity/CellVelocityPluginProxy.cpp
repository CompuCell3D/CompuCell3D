#include <CompuCell3D/plugins/CellVelocity/CellVelocityPlugin.h>

#include <CompuCell3D/Simulator.h>
using namespace CompuCell3D;

#include <BasicUtils/BasicPluginProxy.h>

BasicPluginProxy<Plugin, CellVelocityPlugin>
cellVelocityProxy("CellVelocity", "Tracks cell velocity. Depends on COM plugin",
1, (const char *[]){"CenterOfMass"},
&Simulator::pluginManager);

// cldeque<Coordinates3D<float> >::size_type CellVelocityData::cldequeCapacity=2;
// cldeque<Coordinates3D<float> >::size_type CellVelocityData::enoughDataThreshold=2;
