#include <CompuCell3D/plugins/Velocity/VelocityPlugin.h>

#include <CompuCell3D/Simulator.h>
using namespace CompuCell3D;

#include <BasicUtils/BasicPluginProxy.h>

BasicPluginProxy<Plugin, VelocityPlugin>
velocityProxy("Velocity", "Tracks instantenous cell velocity. Depends on COM plugin",&Simulator::pluginManager);

