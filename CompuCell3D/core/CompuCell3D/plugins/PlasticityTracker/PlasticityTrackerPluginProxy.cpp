#include "PlasticityTrackerPlugin.h"

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/PluginManager.h>

using namespace CompuCell3D;

auto plasticityTrackerPluginProxy = registerPlugin<Plugin, PlasticityTrackerPlugin>(
	"PlasticityTracker", 
	"Initializes and Tracks Plasticity participating cells",
	&Simulator::pluginManager
);
