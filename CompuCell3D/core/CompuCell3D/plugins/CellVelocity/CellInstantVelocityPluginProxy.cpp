#include <CompuCell3D/plugins/CellVelocity/CellInstantVelocityPlugin.h>

#include <CompuCell3D/Simulator.h>
using namespace CompuCell3D;

#include <BasicUtils/BasicPluginProxy.h>

BasicPluginProxy<Plugin, CellInstantVelocityPlugin>
cellInstantVelocityProxy("CellInstantVelocity", "Tracks instantenous cell velocity. Depends on COM plugin",
1, (const char *[]){"CenterOfMass"},
&Simulator::pluginManager);

