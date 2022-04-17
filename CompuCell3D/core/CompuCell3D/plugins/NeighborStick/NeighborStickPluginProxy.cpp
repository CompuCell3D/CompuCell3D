

#include "NeighborStickPlugin.h"

#include <CompuCell3D/Simulator.h>
using namespace CompuCell3D;

#include <BasicUtils/BasicPluginProxy.h>

BasicPluginProxy<Plugin, NeighborStickPlugin>
orientedContactProxy("NeighborStick", "Adds the interaction energy function and orientation.",
	     &Simulator::pluginManager);
