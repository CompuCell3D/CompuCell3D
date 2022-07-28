

#include "VolumeTrackerPlugin.h"

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/PluginManager.h>

using namespace CompuCell3D;

auto volumeTrackerProxy = registerPlugin<Plugin, VolumeTrackerPlugin>(
	"VolumeTracker", 
	"Tracks cell volumes",
	&Simulator::pluginManager
);
