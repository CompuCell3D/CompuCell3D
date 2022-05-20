

#include "NeighborTrackerPlugin.h"

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/PluginManager.h>

using namespace CompuCell3D;

auto NeighborTrackerProxy = registerPlugin<Plugin, NeighborTrackerPlugin>(
	"NeighborTracker", 
	"Tracks cell neighbors and stores them in a list", 
	&Simulator::pluginManager
);
