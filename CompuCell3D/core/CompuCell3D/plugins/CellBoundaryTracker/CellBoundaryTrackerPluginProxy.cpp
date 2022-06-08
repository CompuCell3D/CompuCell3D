

#include "CellBoundaryTrackerPlugin.h"

#include <CompuCell3D/Simulator.h>
using namespace CompuCell3D;

#include <BasicUtils/BasicPluginProxy.h>

BasicPluginProxy<Plugin, CellBoundaryTrackerPlugin> 
CellBoundaryTrackerProxy("CellBoundaryTracker", "Tracks cell boundary and stores cell neighbours in a list",
	    &Simulator::pluginManager);
